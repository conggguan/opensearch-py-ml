# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import argparse
import json
import os
import shutil
import sys
import warnings
from typing import Optional, Tuple

import autotracing_utils
from mdutils.fileutils import MarkDownFile

from opensearch_py_ml.ml_models import NeuralSparseV2Model

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(THIS_DIR, "../..")
sys.path.append(ROOT_DIR)

from opensearch_py_ml.ml_commons import MLCommonClient
from tests import OPENSEARCH_TEST_CLIENT

BOTH_FORMAT = "BOTH"
TORCH_SCRIPT_FORMAT = "TORCH_SCRIPT"
ONNX_FORMAT = "ONNX"

TEMP_MODEL_PATH = "temp_model_path"
TORCHSCRIPT_FOLDER_PATH = "model-torchscript/"
ONNX_FOLDER_PATH = "model-onnx/"
UPLOAD_FOLDER_PATH = "upload/"
MODEL_CONFIG_FILE_NAME = "ml-commons_model_config.json"
OUTPUT_DIR = "trace_output/"
LICENSE_VAR_FILE = "apache_verified.txt"
DESCRIPTION_VAR_FILE = "description.txt"
TEST_SENTENCES = [
    "First test sentence",
    "This is another sentence used for testing model embedding outputs.",
    "OpenSearch is a scalable, flexible, and extensible open-source software suite for search, analytics, and observability applications licensed under Apache 2.0. Powered by Apache Lucene and driven by the OpenSearch Project community, OpenSearch offers a vendor-agnostic toolset you can use to build secure, high-performance, cost-efficient applications. Use OpenSearch as an end-to-end solution or connect it with your preferred open-source tools or partner projects.",
]
RTOL_TEST = 1e-03
ATOL_TEST = 1e-05


def verify_license_in_md_file() -> bool:
    """
    Verify that the model is licensed under Apache 2.0
    by looking at metadata in README.md file of the model

    TODO: Support other open source licenses in future

    :return: Whether the model is licensed under Apache 2.0
    :rtype: Bool
    """
    try:
        readme_data = MarkDownFile.read_file(TEMP_MODEL_PATH + "/" + "README.md")
    except Exception as e:
        print(f"Cannot verify the license: {e}")
        return False

    start = readme_data.find("---")
    end = readme_data.find("---", start + 3)
    if start == -1 or end == -1:
        return False
    metadata_info = readme_data[start + 3 : end]
    if "apache-2.0" in metadata_info.lower():
        print("\nFound apache-2.0 license at " + TEMP_MODEL_PATH + "/README.md")
        return True
    else:
        print("\nDid not find apache-2.0 license at " + TEMP_MODEL_PATH + "/README.md")
        return False


def trace_sparse_encoding_model(
    model_id: str,
    model_version: str,
    model_format: str,
    model_description: Optional[str] = None
) -> Tuple[str, str]:
    """
    Trace the pretrained sentence transformer model, create a model config file,
    and return a path to the model file and a path to the model config file required for model registration

    :param model_id: Model ID of the pretrained model
    :type model_id: string
    :param model_version: Version of the pretrained model for registration
    :type model_version: string
    :param model_format: Model format ("TORCH_SCRIPT" or "ONNX")
    :type model_format: string
    :param model_description: Model description input
    :type model_description: string
    :return: Tuple of model_path (path to model zip file) and model_config_path (path to model config json file)
    :rtype: Tuple[str, str]
    """
    folder_path = (
        TORCHSCRIPT_FOLDER_PATH
        if model_format == TORCH_SCRIPT_FORMAT
        else ONNX_FOLDER_PATH
    )

    # 1.) Initiate a sentence transformer model class object
    pre_trained_model = None
    try:
        pre_trained_model = NeuralSparseV2Model(
            model_id=model_id, folder_path=folder_path, overwrite=True
        )
    except Exception as e:
        assert (
            False
        ), f"Raised Exception in tracing {model_format} model\
                             during initiating a sentence transformer model class object: {e}"

    # 2.) Save the model in the specified format
    model_path = None
    try:
        if model_format == TORCH_SCRIPT_FORMAT:
            model_path = pre_trained_model.save_as_pt(
                model_id=model_id,
                sentences=TEST_SENTENCES,
                add_apache_license=True,
            )
        else:
            model_path = pre_trained_model.save_as_onnx(
                model_id=model_id, add_apache_license=True
            )
    except Exception as e:
        assert False, f"Raised Exception during saving model as {model_format}: {e}"

    # 3.) Create a model config json file
    model_config_path = None
    try:
        model_config_path = pre_trained_model.make_model_config_json(
            version_number=model_version,
            model_format=model_format,
            description=model_description
        )
    except Exception as e:
        assert (
            False
        ), f"Raised Exception during making model config file for {model_format} model: {e}"

    # 4.) Preview model config
    print(f"\n+++++ {model_format} Model Config +++++\n")
    with open(model_config_path, "r") as f:
        model_config = json.load(f)
        print(json.dumps(model_config, indent=4))
    print("\n+++++++++++++++++++++++++++++++++++++++\n")

    # 5.) Return model_path & model_config_path for model registration
    return model_path, model_config_path


def register_and_deploy_sentence_transformer_model(
    ml_client: "MLCommonClient",
    model_path: str,
    model_config_path: str,
    model_format: str,
    query: list[str],
) -> dict:
    encoding_data = None
    model_id = autotracing_utils.register_and_deploy_model(
        ml_client, model_format, model_path, model_config_path
    )
    autotracing_utils.check_model_status(ml_client, model_id, model_format)
    try:
        encoding_output = ml_client.generate_sparse_encoding(model_id, query)
        # assert len(encoding_output.get("inference_results")) == len(TEST_SENTENCES)
        encoding_data = encoding_output["inference_results"][0]["output"][0][
            "dataAsMap"
        ]["response"][0]

    except Exception as e:
        assert (
            False
        ), f"Raised Exception in generating sentence embedding with {model_format} model: {e}"
    autotracing_utils.undeploy_model(ml_client, model_id, model_format)
    autotracing_utils.delete_model(ml_client, model_id, model_format)
    return encoding_data


def verify_sparse_encoding(
    original_embedding_data: dict,
    tracing_embedding_data: dict,
) -> bool:

    if original_embedding_data.keys() != tracing_embedding_data.keys():
        print("Different encoding dimensions")
        return False
    tolerance = 0.01
    for key in original_embedding_data:
        if abs(original_embedding_data[key] - tracing_embedding_data[key]) > tolerance:
            print(
                f"{key}'s score has gap: {original_embedding_data[key]} != {tracing_embedding_data[key]}"
            )
            return False
    return True


def prepare_files_for_uploading(
    model_id: str,
    model_version: str,
    model_format: str,
    src_model_path: str,
    src_model_config_path: str,
) -> tuple[str, str]:
    """
    Prepare files for uploading by storing them in UPLOAD_FOLDER_PATH

    :param model_id: Model ID of the pretrained model
    :type model_id: string
    :param model_version: Version of the pretrained model for registration
    :type model_version: string
    :param model_format: Model format ("TORCH_SCRIPT" or "ONNX")
    :type model_format: string
    :param src_model_path: Path to model files for uploading
    :type src_model_path: string
    :param src_model_config_path: Path to model config files for uploading
    :type src_model_config_path: string
    :return: Tuple of dst_model_path (path to model zip file) and dst_model_config_path
    (path to model config json file) in the UPLOAD_FOLDER_PATH
    :rtype: Tuple[str, str]
    """
    model_type, model_name = model_id.split("/")
    model_format = model_format.lower()
    folder_to_delete = (
        TORCHSCRIPT_FOLDER_PATH if model_format == "torch_script" else ONNX_FOLDER_PATH
    )

    # Store to be uploaded files in UPLOAD_FOLDER_PATH
    try:
        dst_model_dir = (
            f"{UPLOAD_FOLDER_PATH}{model_name}/{model_version}/{model_format}"
        )
        os.makedirs(dst_model_dir, exist_ok=True)
        dst_model_filename = (
            f"{model_type}_{model_name}-{model_version}-{model_format}.zip"
        )
        dst_model_path = dst_model_dir + "/" + dst_model_filename
        shutil.copy(src_model_path, dst_model_path)
        print(f"\nCopied {src_model_path} to {dst_model_path}")

        dst_model_config_dir = (
            f"{UPLOAD_FOLDER_PATH}{model_name}/{model_version}/{model_format}"
        )
        os.makedirs(dst_model_config_dir, exist_ok=True)
        dst_model_config_filename = "config.json"
        dst_model_config_path = dst_model_config_dir + "/" + dst_model_config_filename
        shutil.copy(src_model_config_path, dst_model_config_path)
        print(f"Copied {src_model_config_path} to {dst_model_config_path}")
    except Exception as e:
        assert (
            False
        ), f"Raised Exception during preparing {model_format} files for uploading: {e}"

    # Delete model folder downloaded from HuggingFace during model tracing
    try:
        shutil.rmtree(folder_to_delete)
    except Exception as e:
        assert False, f"Raised Exception while deleting {folder_to_delete}: {e}"

    return dst_model_path, dst_model_config_path


def store_license_verified_variable(license_verified: bool) -> None:
    """
    Store whether the model is licensed under Apache 2.0 in OUTPUT_DIR/LICENSE_VAR_FILE
    to be used to generate issue body for manual approval

    :param license_verified: Whether the model is licensed under Apache 2.0
    :return: No return value expected
    :rtype: None
    """
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        license_var_filepath = OUTPUT_DIR + "/" + LICENSE_VAR_FILE
        with open(license_var_filepath, "w") as f:
            f.write(str(license_verified))
    except Exception as e:
        print(
            f"Cannot store license_verified ({license_verified}) in {license_var_filepath}: {e}"
        )


def store_description_variable(config_path_for_checking_description: str) -> None:
    """
    Store model description in OUTPUT_DIR/DESCRIPTION_VAR_FILE
    to be used to generate issue body for manual approval

    :param config_path_for_checking_description: Path to config json file
    :type config_path_for_checking_description: str
    :return: No return value expected
    :rtype: None
    """
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        description_var_filepath = OUTPUT_DIR + "/" + DESCRIPTION_VAR_FILE
        with open(config_path_for_checking_description, "r") as f:
            config_dict = json.load(f)
            description = (
                config_dict["description"] if "description" in config_dict else "-"
            )
        print(f"Storing the following description at {description_var_filepath}")
        print(description)
        with open(description_var_filepath, "w") as f:
            f.write(description)
    except Exception as e:
        print(
            f"Cannot store description ({description}) in {description_var_filepath}: {e}"
        )


def main(
    model_id: str,
    model_version: str,
    tracing_format: str,
    model_description: Optional[str] = None,
) -> None:
    """
    Perform model auto-tracing and prepare files for uploading to OpenSearch model hub

    :param model_id: Model ID of the pretrained model
    :type model_id: string
    :param model_version: Version of the pretrained model for registration
    :type model_version: string
    :param tracing_format: Tracing format ("TORCH_SCRIPT", "ONNX", or "BOTH")
    :type tracing_format: string
    :param model_description: Model description input
    :type model_description: string
    :return: No return value expected
    :rtype: None
    """
    TEST_SENTENCES = ["Nice to meet you!"]

    print(
        f"""
    === Begin running model_autotracing.py ===
    Model ID: {model_id}
    Model Version: {model_version}
    Tracing Format: {tracing_format}
    Model Description: {model_description if model_description is not None else 'N/A'}
    ==========================================
    """
    )

    ml_client = MLCommonClient(OPENSEARCH_TEST_CLIENT)
    pre_trained_model = NeuralSparseV2Model(model_id)
    original_encoding_data = pre_trained_model.process_queries(TEST_SENTENCES)
    pre_trained_model.save(path=TEMP_MODEL_PATH)
    license_verified = verify_license_in_md_file()

    try:
        shutil.rmtree(TEMP_MODEL_PATH)
    except Exception as e:
        assert False, f"Raised Exception while deleting {TEMP_MODEL_PATH}: {e}"

    if tracing_format in [TORCH_SCRIPT_FORMAT, BOTH_FORMAT]:
        print("--- Begin tracing a model in TORCH_SCRIPT ---")
        (
            torchscript_model_path,
            torchscript_model_config_path,
        ) = trace_sparse_encoding_model(
            model_id, model_version, TORCH_SCRIPT_FORMAT, model_description=None
        )

        torchscript_encoding_data = register_and_deploy_sentence_transformer_model(
            ml_client,
            torchscript_model_path,
            torchscript_model_config_path,
            TORCH_SCRIPT_FORMAT,
            TEST_SENTENCES,
        )

        pass_test = verify_sparse_encoding(
            original_encoding_data, torchscript_encoding_data
        )
        assert (
            pass_test
        ), f"Failed while verifying embeddings of {model_id} model in TORCH_SCRIPT format"

        (
            torchscript_dst_model_path,
            torchscript_dst_model_config_path,
        ) = prepare_files_for_uploading(
            model_id,
            model_version,
            TORCH_SCRIPT_FORMAT,
            torchscript_model_path,
            torchscript_model_config_path,
        )

        config_path_for_checking_description = torchscript_dst_model_config_path
        print("--- Finished tracing a model in TORCH_SCRIPT ---")

    if tracing_format in [ONNX_FORMAT, BOTH_FORMAT]:
        print("--- Begin tracing a model in ONNX ---")
        (
            onnx_model_path,
            onnx_model_config_path,
        ) = trace_sparse_encoding_model(
            model_id, model_version, ONNX_FORMAT, model_description=None
        )

        onnx_embedding_data = register_and_deploy_sentence_transformer_model(
            ml_client,
            onnx_model_path,
            onnx_model_config_path,
            ONNX_FORMAT,
            TEST_SENTENCES,
        )

        pass_test = verify_sparse_encoding(original_encoding_data, onnx_embedding_data)
        assert (
            pass_test
        ), f"Failed while verifying embeddings of {model_id} model in ONNX format"

        onnx_dst_model_path, onnx_dst_model_config_path = prepare_files_for_uploading(
            model_id,
            model_version,
            ONNX_FORMAT,
            onnx_model_path,
            onnx_model_config_path,
        )

        config_path_for_checking_description = onnx_dst_model_config_path
        print("--- Finished tracing a model in ONNX ---")

    store_license_verified_variable(license_verified)
    store_description_variable(config_path_for_checking_description)

    print("\n=== Finished running model_autotracing.py ===")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="Unverified HTTPS request")
    warnings.filterwarnings("ignore", message="TracerWarning: torch.tensor")
    warnings.filterwarnings(
        "ignore", message="using SSL with verify_certs=False is insecure."
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_id",
        type=str,
        help="Model ID for auto-tracing and uploading (e.g. sentence-transformers/msmarco-distilbert-base-tas-b)",
    )
    parser.add_argument(
        "model_version", type=str, help="Model version number (e.g. 1.0.1)"
    )
    parser.add_argument(
        "tracing_format",
        choices=["BOTH", "TORCH_SCRIPT", "ONNX"],
        help="Model format for auto-tracing",
    )
    parser.add_argument(
        "-md",
        "--model_description",
        type=str,
        nargs="?",
        default=None,
        const=None,
        help="Model description if you want to overwrite the default description",
    )
    args = parser.parse_args()

    main(
        args.model_id, args.model_version, args.tracing_format
    )
