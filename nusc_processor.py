import os
import json
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from tqdm import tqdm
from typing import Dict, List


class NuScenesProcessor:
    def __init__(self, dataroot: str = 'C:\TUSHAR FINAL YEAR PROJECT', version: str = 'v1.0-mini'):
        self.nusc = NuScenes(version=version, dataroot=dataroot)
        self.scene_splits = create_splits_scenes()

    def process_dataset(self, output_dir: str = './data/processed'):
        """Process entire dataset into training-friendly format"""
        os.makedirs(output_dir, exist_ok=True)

        samples = self._get_all_samples()
        data_records = []

        for sample in tqdm(samples, desc="Processing samples"):
            record = self._process_sample(sample)
            data_records.append(record)

            # Save individual sample
            with open(f"{output_dir}/sample_{sample['token']}.json", 'w') as f:
                json.dump(record, f)

        # Save complete dataframe
        return data_records

    def _get_all_samples(self) -> List[Dict]:
        """Get all samples from dataset"""
        samples = []
        for scene in self.nusc.scene:
            sample_token = scene['first_sample_token']
            while sample_token:
                sample = self.nusc.get('sample', sample_token)
                samples.append(sample)
                sample_token = sample['next']
        return samples

    def _process_sample(self, sample: Dict) -> Dict:
        """Process a single sample into training format"""
        # Get camera data (focus on front camera)
        cam_token = sample['data']['CAM_FRONT']
        cam_data = self.nusc.get('sample_data', cam_token)
        img_path = os.path.join(self.nusc.dataroot, cam_data['filename'])

        # Get annotations
        annotations = []
        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            annotations.append({
                'category': ann['category_name'],
                'bbox': self.nusc.get_box(ann_token),
                'attributes': [self.nusc.get('attribute', attr_token)['name']
                               for attr_token in ann['attribute_tokens']]
            })

        # Get scene description (from human annotations if available)
        description = self._get_scene_description(sample)

        return {
            'sample_token': sample['token'],
            'image_path': img_path,
            'annotations': annotations,
            'description': description,
            'timestamp': cam_data['timestamp']
        }

    def _get_scene_description(self, sample: Dict) -> str:
        """Generate human-like scene description"""
        # Try to get human annotation first
        if 'description' in sample:
            return sample['description']

        # Fallback to automatic description
        anns = [self.nusc.get('sample_annotation', t) for t in sample['anns']]
        objects = [self.nusc.get('category', a['category_token'])['name'] for a in anns]

        unique_objects = []
        for obj in set(objects):
            count = objects.count(obj)
            unique_objects.append(f"{count} {obj}{'s' if count > 1 else ''}")

        return f"Scene contains: {', '.join(unique_objects)}"
