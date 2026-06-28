#!/usr/bin/env python3
"""
Generates a mock ZIP archive containing multiple DICOM MR image files (.dcm) for testing.
"""

import os
import tempfile
import zipfile
from pathlib import Path
import numpy as np
import pydicom
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

def create_mock_dicom(filename, pixel_data, slice_idx, slice_location, patient_name="Test^Patient", patient_id="123456", study_uid=None, series_uid=None):
    # Populate required values for file meta information
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4' # MR Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = '1.2.3.4'

    # Create the FileDataset instance
    ds = pydicom.FileDataset(filename, {}, file_meta=file_meta)
    
    # Add patient & study attributes
    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.StudyInstanceUID = study_uid if study_uid else generate_uid()
    ds.SeriesInstanceUID = series_uid if series_uid else generate_uid()
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "MR"
    
    # Add matrix grid properties
    ds.Rows, ds.Columns = pixel_data.shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 1.0
    ds.ImagePositionPatient = [0.0, 0.0, float(slice_location)]
    ds.SliceLocation = float(slice_location)
    ds.InstanceNumber = int(slice_idx)
    
    # Convert pixel data to int16 bytes
    ds.PixelData = pixel_data.astype(np.uint16).tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    
    ds.save_as(filename)

def build_mock_zip(output_path="./data/mock_dicom_volume.zip"):
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    temp_dir = Path(tempfile.mkdtemp())
    try:
        modalities = ["T1", "T1ce", "T2", "FLAIR"]
        patient_name = "Smith^John"
        patient_id = "MR-98765"
        study_uid = generate_uid()
        
        # Create folder structure for each modality
        for mod in modalities:
            mod_dir = temp_dir / mod
            mod_dir.mkdir(parents=True, exist_ok=True)
            series_uid = generate_uid()
            
            # Create 5 slices per modality
            for i in range(5):
                # Create a 240x240 slice with a circle of values (slightly offset by modality)
                slice_data = np.zeros((240, 240), dtype=np.uint16)
                x, y = np.ogrid[:240, :240]
                dist_sq = (x - 120)**2 + (y - 120)**2
                
                # Make the circle size/intensity vary by modality for easy visual confirmation
                radius = 50 + modalities.index(mod) * 5
                slice_data[dist_sq <= radius**2] = 400 + i * 20 + modalities.index(mod) * 100
                
                dcm_filename = mod_dir / f"slice_{i:03d}.dcm"
                create_mock_dicom(
                    str(dcm_filename), 
                    slice_data, 
                    i, 
                    float(i * 1.5),
                    patient_name=patient_name,
                    patient_id=patient_id,
                    study_uid=study_uid,
                    series_uid=series_uid
                )
                
        # Zip folders recursively
        with zipfile.ZipFile(output_path, 'w') as zip_ref:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = Path(root) / file
                    relative_path = file_path.relative_to(temp_dir)
                    zip_ref.write(str(file_path), arcname=str(relative_path))
                
        print(f"Mock multi-modal DICOM zip successfully created at {output_path}")
    finally:
        # Cleanup temp directory
        import shutil
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    build_mock_zip()
