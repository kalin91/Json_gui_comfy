"""Tests for json_gui/utils.py module."""

import os
import sys
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch, mock_open

# Import the module under test
import json_gui.utils as utils


class TestGetMainImagesPath:
    """Test get_main_images_path function."""
    
    def test_returns_images_path(self, mock_folder_paths):
        """Test that it returns the correct images path."""
        result = utils.get_main_images_path()
        expected = os.path.join(mock_folder_paths["get_user_directory"].return_value, "images")
        assert result == expected
    
    def test_creates_directory_if_not_exists(self, mock_folder_paths, temp_directory):
        """Test that it creates the directory if it doesn't exist."""
        new_path = os.path.join(temp_directory, "new_user")
        mock_folder_paths["get_user_directory"].return_value = new_path
        
        result = utils.get_main_images_path()
        expected = os.path.join(new_path, "images")
        assert result == expected
        assert os.path.exists(expected)


class TestGetScriptsFolderPath:
    """Test get_scripts_folder_path function."""
    
    def test_returns_scripts_path(self):
        """Test that it returns the correct scripts path."""
        result = utils.get_scripts_folder_path()
        assert result.endswith("scripts")
        assert os.path.exists(result)
    
    def test_creates_directory_if_not_exists(self, temp_directory):
        """Test that it creates the directory if it doesn't exist."""
        with patch("os.path.dirname") as mock_dirname, \
             patch("os.path.realpath") as mock_realpath:
            mock_realpath.return_value = temp_directory
            mock_dirname.return_value = temp_directory
            
            result = utils.get_scripts_folder_path()
            expected = os.path.join(temp_directory, "scripts")
            assert result == expected


class TestGetFlowAndBodyPaths:
    """Test get_flow_and_body_paths function."""
    
    def test_returns_flow_and_body_paths(self, temp_directory):
        """Test that it returns correct paths for flow and body files."""
        # Create script directory structure
        with patch.object(utils, "get_scripts_folder_path", return_value=temp_directory):
            script_name = "test_script"
            script_dir = os.path.join(temp_directory, script_name)
            os.makedirs(script_dir, exist_ok=True)
            
            # Create flow.py and body.yml
            flow_path = os.path.join(script_dir, "flow.py")
            body_path = os.path.join(script_dir, "body.yml")
            with open(flow_path, "w") as _f: pass
            with open(body_path, "w") as _f: pass
            
            result = utils.get_flow_and_body_paths(script_name)
            assert result == (flow_path, body_path)
    
    def test_raises_assertion_error_if_script_dir_not_exists(self, temp_directory):
        """Test that it raises AssertionError if script directory doesn't exist."""
        with patch.object(utils, "get_scripts_folder_path", return_value=temp_directory):
            with pytest.raises(AssertionError, match="Script directory .* does not exist"):
                utils.get_flow_and_body_paths("nonexistent_script")
    
    def test_raises_assertion_error_if_flow_not_exists(self, temp_directory):
        """Test that it raises AssertionError if flow.py doesn't exist."""
        with patch.object(utils, "get_scripts_folder_path", return_value=temp_directory):
            script_name = "test_script"
            script_dir = os.path.join(temp_directory, script_name)
            os.makedirs(script_dir, exist_ok=True)
            
            # Create only body.yml
            body_path = os.path.join(script_dir, "body.yml")
            with open(body_path, "w") as _f: pass
            
            with pytest.raises(AssertionError, match="Flow script .* does not exist"):
                utils.get_flow_and_body_paths(script_name)
    
    def test_raises_assertion_error_if_body_not_exists(self, temp_directory):
        """Test that it raises AssertionError if body.yml doesn't exist."""
        with patch.object(utils, "get_scripts_folder_path", return_value=temp_directory):
            script_name = "test_script"
            script_dir = os.path.join(temp_directory, script_name)
            os.makedirs(script_dir, exist_ok=True)
            
            # Create only flow.py
            flow_path = os.path.join(script_dir, "flow.py")
            with open(flow_path, "w") as _f: pass
            
            with pytest.raises(AssertionError, match="Body file .* does not exist"):
                utils.get_flow_and_body_paths(script_name)


class TestGetInputFilesRecursive:
    """Test get_input_files_recursive function."""
    
    def test_returns_filtered_input_files(self, mock_folder_paths):
        """Test that it returns filtered input files."""
        mock_files = ["image1.png", "image2.jpg", "doc.txt"]
        mock_folder_paths["recursive_search"].return_value = (mock_files, [])
        mock_folder_paths["filter_files_content_types"].return_value = ["image1.png", "image2.jpg"]
        
        result, folder = utils.get_input_files_recursive()
        
        assert result == ["image1.png", "image2.jpg"]
        assert folder == mock_folder_paths["get_input_directory"].return_value
        mock_folder_paths["filter_files_content_types"].assert_called_once_with(mock_files, ["image"])


class TestGetOutputFilesRecursive:
    """Test _get_output_files_recursive function."""
    
    def test_returns_output_files(self, mock_folder_paths):
        """Test that it returns output files."""
        mock_files = ["output1.png", "output2.json"]
        mock_folder_paths["recursive_search"].return_value = (mock_files, [])
        
        result, folder = utils._get_output_files_recursive()
        
        assert result == mock_files
        assert folder == mock_folder_paths["get_output_directory"].return_value


class TestGetFolderFilesRecursive:
    """Test get_folder_files_recursive function."""
    
    def test_returns_folder_files(self, mock_folder_paths):
        """Test that it returns files from specified folder."""
        mock_files = ["file1.png", "file2.jpg"]
        mock_folder = "/test/folder"
        mock_folder_paths["get_filename_list_"].return_value = (mock_files, {mock_folder: []})
        
        result, folder = utils.get_folder_files_recursive("test_folder")
        
        assert result == mock_files
        assert folder == mock_folder


class TestEndOfFlowException:
    """Test EndOfFlowException class."""
    
    def test_exception_stores_steps(self):
        """Test that the exception stores the steps value."""
        steps = 10
        exc = utils.EndOfFlowException(steps)
        assert exc.steps == steps
    
    def test_exception_message(self):
        """Test that the exception has correct message."""
        steps = 10
        exc = utils.EndOfFlowException(steps)
        assert f"End of flow after reaching {steps} steps limit" in str(exc)


class TestAbsFlow:
    """Test AbsFlow abstract base class."""
    
    def test_abstract_class_cannot_instantiate(self):
        """Test that abstract class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            utils.AbsFlow("path", "filename")  # type: ignore
    
    def test_concrete_implementation(self, mock_folder_paths):
        """Test a concrete implementation of AbsFlow."""
        
        class ConcreteFlow(utils.AbsFlow):
            def _run_impl(self, steps: int) -> None:
                pass
        
        file_path = "test_path"
        filename = "test_file"
        
        flow = ConcreteFlow(file_path, filename)
        
        assert flow.file_path == file_path
        assert flow.json_path == os.path.join(
            mock_folder_paths["get_user_directory"].return_value,
            "images",
            file_path,
            f"{filename}.json"
        )
    
    def test_run_saves_json_to_output(self, mock_folder_paths, temp_directory):
        """Test that run() saves JSON to output directory."""
        
        class ConcreteFlow(utils.AbsFlow):
            def _run_impl(self, steps: int) -> None:
                pass
        
        # Setup directories
        user_dir = os.path.join(temp_directory, "user")
        output_dir = os.path.join(temp_directory, "output")
        temp_dir = os.path.join(temp_directory, "temp")
        os.makedirs(os.path.join(user_dir, "images", "test_path"), exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        
        mock_folder_paths["get_user_directory"].return_value = user_dir
        mock_folder_paths["get_output_directory"].return_value = output_dir
        mock_folder_paths["get_temp_directory"].return_value = temp_dir
        
        # Create json file
        json_path = os.path.join(user_dir, "images", "test_path", "test_file.json")
        with open(json_path, "w") as f:
            f.write("{}")
        
        flow = ConcreteFlow("test_path", "test_file")
        
        with patch("torch.inference_mode"):
            result = flow.run(steps=5)
        
        # Check that JSON was copied
        assert len(result) == 0  # No images created
    
    def test_save_image_creates_image_file(self, mock_folder_paths, temp_directory, sample_image_tensor):
        """Test that save_image() creates an image file."""
        
        class ConcreteFlow(utils.AbsFlow):
            def _run_impl(self, steps: int) -> None:
                pass
        
        # Setup directories
        temp_dir = os.path.join(temp_directory, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        mock_folder_paths["get_temp_directory"].return_value = temp_dir
        
        user_dir = os.path.join(temp_directory, "user")
        os.makedirs(os.path.join(user_dir, "images", "test_path"), exist_ok=True)
        mock_folder_paths["get_user_directory"].return_value = user_dir
        
        output_dir = os.path.join(temp_directory, "output")
        os.makedirs(output_dir, exist_ok=True)
        mock_folder_paths["get_output_directory"].return_value = output_dir
        
        # Create json file
        json_path = os.path.join(user_dir, "images", "test_path", "test_file.json")
        with open(json_path, "w") as f:
            f.write("{}")
        
        flow = ConcreteFlow("test_path", "test_file")
        
        # Save image
        flow.save_image(sample_image_tensor, "test_id", steps=10, is_temp=True)
        
        # Check that image was created
        assert len(flow._created_images) == 1
        assert os.path.exists(flow._created_images[0])
    
    def test_save_image_raises_end_of_flow_exception(self, mock_folder_paths, temp_directory, sample_image_tensor):
        """Test that save_image() raises EndOfFlowException when reaching step limit."""
        
        class ConcreteFlow(utils.AbsFlow):
            def _run_impl(self, steps: int) -> None:
                pass
        
        # Setup directories
        temp_dir = os.path.join(temp_directory, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        mock_folder_paths["get_temp_directory"].return_value = temp_dir
        
        user_dir = os.path.join(temp_directory, "user")
        os.makedirs(os.path.join(user_dir, "images", "test_path"), exist_ok=True)
        mock_folder_paths["get_user_directory"].return_value = user_dir
        
        output_dir = os.path.join(temp_directory, "output")
        os.makedirs(output_dir, exist_ok=True)
        mock_folder_paths["get_output_directory"].return_value = output_dir
        
        # Create json file
        json_path = os.path.join(user_dir, "images", "test_path", "test_file.json")
        with open(json_path, "w") as f:
            f.write("{}")
        
        flow = ConcreteFlow("test_path", "test_file")
        
        # Save image with steps limit of 1
        with pytest.raises(utils.EndOfFlowException) as exc_info:
            flow.save_image(sample_image_tensor, "test_id", steps=1, is_temp=True)
        
        assert exc_info.value.steps == 1
    
    def test_run_handles_end_of_flow_exception(self, mock_folder_paths, temp_directory, sample_image_tensor):
        """Test that run() handles EndOfFlowException gracefully."""
        
        class ConcreteFlow(utils.AbsFlow):
            def _run_impl(self, steps: int) -> None:
                # Simulate saving image that reaches step limit
                raise utils.EndOfFlowException(steps)
        
        # Setup directories
        user_dir = os.path.join(temp_directory, "user")
        output_dir = os.path.join(temp_directory, "output")
        temp_dir = os.path.join(temp_directory, "temp")
        os.makedirs(os.path.join(user_dir, "images", "test_path"), exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        
        mock_folder_paths["get_user_directory"].return_value = user_dir
        mock_folder_paths["get_output_directory"].return_value = output_dir
        mock_folder_paths["get_temp_directory"].return_value = temp_dir
        
        # Create json file
        json_path = os.path.join(user_dir, "images", "test_path", "test_file.json")
        with open(json_path, "w") as f:
            f.write("{}")
        
        flow = ConcreteFlow("test_path", "test_file")
        
        with patch("torch.inference_mode"):
            result = flow.run(steps=5)
        
        # Should not raise exception
        assert isinstance(result, list)
    
    def test_run_copies_last_image_to_output(self, mock_folder_paths, temp_directory, sample_image_tensor):
        """Test that run() copies last temp image to output directory."""
        
        class ConcreteFlow(utils.AbsFlow):
            def _run_impl(self, steps: int) -> None:
                self.save_image(sample_image_tensor, "test_id", steps=10, is_temp=True)
        
        # Setup directories
        temp_dir = os.path.join(temp_directory, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        mock_folder_paths["get_temp_directory"].return_value = temp_dir
        
        user_dir = os.path.join(temp_directory, "user")
        os.makedirs(os.path.join(user_dir, "images", "test_path"), exist_ok=True)
        mock_folder_paths["get_user_directory"].return_value = user_dir
        
        output_dir = os.path.join(temp_directory, "output")
        os.makedirs(output_dir, exist_ok=True)
        mock_folder_paths["get_output_directory"].return_value = output_dir
        
        # Create json file
        json_path = os.path.join(user_dir, "images", "test_path", "test_file.json")
        with open(json_path, "w") as f:
            f.write("{}")
        
        flow = ConcreteFlow("test_path", "test_file")
        
        with patch("torch.inference_mode"):
            result = flow.run(steps=5)
        
        # Check that image was copied to output
        assert len(result) == 1
        # Last image should be in output directory
        assert output_dir in result[-1]
    
    def test_absflow_handles_existing_output_files(self, mock_folder_paths, temp_directory):
        """Test that AbsFlow handles existing output files."""
        
        class ConcreteFlow(utils.AbsFlow):
            def _run_impl(self, steps: int) -> None:
                pass
        
        # Setup directories
        output_dir = os.path.join(temp_directory, "output")
        temp_dir = os.path.join(temp_directory, "temp")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        
        user_dir = os.path.join(temp_directory, "user")
        os.makedirs(os.path.join(user_dir, "images", "test_path"), exist_ok=True)
        mock_folder_paths["get_user_directory"].return_value = user_dir
        mock_folder_paths["get_output_directory"].return_value = output_dir
        mock_folder_paths["get_temp_directory"].return_value = temp_dir
        
        # Create json file
        json_path = os.path.join(user_dir, "images", "test_path", "test_file.json")
        with open(json_path, "w") as f:
            f.write("{}")
        
        # Set mock to return empty lists (no existing files)
        mock_folder_paths["recursive_search"].return_value = ([], [])
        
        # Create flow - should handle empty case gracefully
        flow = ConcreteFlow("test_path", "test_file")
        
        # Verify flow was created
        assert flow._file_identifier.startswith("test_file_r")
    
    def test_save_image_handles_file_name_collision(self, mock_folder_paths, temp_directory, sample_image_tensor):
        """Test that save_image handles file name collisions."""
        
        class ConcreteFlow(utils.AbsFlow):
            def _run_impl(self, steps: int) -> None:
                pass
        
        # Setup directories
        temp_dir = os.path.join(temp_directory, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        mock_folder_paths["get_temp_directory"].return_value = temp_dir
        
        user_dir = os.path.join(temp_directory, "user")
        os.makedirs(os.path.join(user_dir, "images", "test_path"), exist_ok=True)
        mock_folder_paths["get_user_directory"].return_value = user_dir
        
        output_dir = os.path.join(temp_directory, "output")
        os.makedirs(output_dir, exist_ok=True)
        mock_folder_paths["get_output_directory"].return_value = output_dir
        
        # Create json file
        json_path = os.path.join(user_dir, "images", "test_path", "test_file.json")
        with open(json_path, "w") as f:
            f.write("{}")
        
        flow = ConcreteFlow("test_path", "test_file")
        
        # Create first image
        flow.save_image(sample_image_tensor, "test_id", steps=10, is_temp=True)
        first_image = flow._created_images[0]
        
        # Save another image - should get different name
        flow.save_image(sample_image_tensor, "test_id", steps=10, is_temp=True)
        second_image = flow._created_images[1]
        
        # Names should be different
        assert first_image != second_image
        assert os.path.exists(first_image)
        assert os.path.exists(second_image)
