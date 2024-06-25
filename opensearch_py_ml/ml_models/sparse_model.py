# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import os
from zipfile import ZipFile

import requests

LICENSE_URL = "https://github.com/opensearch-project/opensearch-py-ml/raw/main/LICENSE"


class SparseModel:

    def __init__(
        self,
        model_id: str,
        folder_path: str = "./model_files/",
        overwrite: bool = False,
    ):
        self.model_id = model_id
        self.folder_path = folder_path
        self.overwrite = overwrite

    def pro_process(self):
        pass

    def post_process(self):
        pass

    def _add_apache_license_to_zip(self, zip_file_path: str):
        with ZipFile(zip_file_path, "a") as zipObj, requests.get(LICENSE_URL) as r:
            assert r.status_code == 200, "Failed to download license"
            zipObj.writestr("LICENSE", r.content)

    def zip_model(
        self,
        model_name: str = None,
        zip_file_name: str = None,
        add_apache_license: bool = False,
    ):
        model_name = model_name or f"{self.model_id.split('/')[-1]}.pt"
        zip_file_name = zip_file_name or f"{self.model_id.split('/')[-1]}.zip"

        model_path = os.path.join(self.folder_path, model_name)
        zip_file_path = os.path.join(self.folder_path, zip_file_name)
        tokenizer_json_path = os.path.join(self.folder_path, "tokenizer.json")

        if not all(map(os.path.exists, [model_path, tokenizer_json_path])):
            raise FileNotFoundError("Required files not found for zipping.")

        with ZipFile(zip_file_path, "w") as zipObj:
            zipObj.write(model_path, arcname=model_name)
            zipObj.write(tokenizer_json_path, arcname="tokenizer.json")

        if add_apache_license:
            self._add_apache_license_to_zip(zip_file_path)

    def save_as_pt(
        self, sentences: [str], model_name: str = None, add_apache_license: bool = False
    ):
        pass

    def make_model_config_json(
        self, version_number: str = "1.0", model_format: str = "TORCH_SCRIPT"
    ) -> str:
        pass

    def save(self, path: str):
        pass

    def _fill_null_truncation_field(
        self,
        save_json_folder_path: str,
        max_length: int,
    ) -> None:
        """
        Fill truncation field in tokenizer.json when it is null

        :param save_json_folder_path:
             path to save model json file, e.g, "home/save_pre_trained_model_json/")
        :type save_json_folder_path: string
        :param max_length:
             maximum sequence length for model
        :type max_length: int
        :return: no return value expected
        :rtype: None
        """
        tokenizer_file_path = os.path.join(save_json_folder_path, "tokenizer.json")
        with open(tokenizer_file_path) as user_file:
            parsed_json = json.load(user_file)
        if "truncation" not in parsed_json or parsed_json["truncation"] is None:
            parsed_json["truncation"] = {
                "direction": "Right",
                "max_length": max_length,
                "strategy": "LongestFirst",
                "stride": 0,
            }
            with open(tokenizer_file_path, "w") as file:
                json.dump(parsed_json, file, indent=2)

    def _add_apache_license_to_model_zip_file(self, model_zip_file_path: str):
        """
        Add Apache-2.0 license file to the model zip file at model_zip_file_path

        :param model_zip_file_path:
            Path to the model zip file
        :type model_zip_file_path: string
        :return: no return value expected
        :rtype: None
        """
        r = requests.get(LICENSE_URL)
        assert r.status_code == 200, "Failed to add license file to the model zip file"

        with ZipFile(str(model_zip_file_path), "a") as zipObj:
            zipObj.writestr("LICENSE", r.content)
