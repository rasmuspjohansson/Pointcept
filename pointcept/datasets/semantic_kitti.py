import os
import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class SemanticKITTIDataset(DefaultDataset):
    def __init__(self, ignore_index=1, **kwargs):
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        self.learning_map_inv = self.get_learning_map_inv(ignore_index)
        super().__init__(ignore_index=ignore_index, **kwargs)

    def get_data_list(self):
        split2seq = dict(
            train=["00"],# '00_split' #train=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
            val=["00"], #val=[8],
            test =["00"], #test=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        )
        if isinstance(self.split, str):
            seq_list = split2seq[self.split]
        elif isinstance(self.split, list):
            seq_list = []
            for split in self.split:
                seq_list += split2seq[split]
        else:
            raise NotImplementedError

        data_list = []
        for seq in seq_list:
            seq = str(seq).zfill(2)
            seq_folder = os.path.join(self.data_root, "dataset", "sequences", seq)
            seq_files = sorted(os.listdir(os.path.join(seq_folder, "velodyne")))
            data_list += [
                os.path.join(seq_folder, "velodyne", file) for file in seq_files
            ]
        return data_list

    # Compact version for quick testing:
    def get_dummy_data(self, idx):
        """Minimal dummy data generator (worked when name get_data )"""
        np.random.seed(idx)
        max_num = 25 # this number casues the cuda error ! 4608015 # this number is ok 25
        coord = np.random.uniform(-max_num, max_num, (6000, 3)).astype(np.float32)
        strength = np.random.uniform(0, 1, (6000, 1)).astype(np.float32)
        # Create segments with all classes represented
        number = 12 #(was 14)
        segment = np.random.randint(0, number, 6000).astype(np.int32)
        # Ensure we have all classes 0-13
        segment[:number] = np.arange(number)
        # Add some ignore samples
        segment[number:114] = self.ignore_index  # 100 ignore samples
        # Shuffle
        np.random.shuffle(segment)
    
        return dict(
            coord=coord,
            strength=strength, 
            segment=segment,
            name=f"dummy_{idx:06d}"
        )


    def get_data(self, idx):
        print(idx)
        data_path = self.data_list[idx % len(self.data_list)]
        print("data_path"+str(data_path))
        with open(data_path, "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3].astype(np.float32)
        # now done in separate trANSFOMR ! coord -= coord.mean(axis=0)
        #coord /= np.abs(coord).max()
        print(coord[0])
        print(coord.min())
        print(coord.max())


        strength = scan[:, -1].reshape([-1, 1]).astype(np.float32)
        label_file = data_path.replace("velodyne", "labels").replace(".bin", ".label")
        if os.path.exists(label_file):
            with open(label_file, "rb") as a:
                raw_segment = np.fromfile(a, dtype=np.int32).reshape(-1)
                segment = np.vectorize(self.learning_map.__getitem__)(
                    raw_segment & 0xFFFF
                ).astype(np.int32)
        else:
            segment = np.zeros(scan.shape[0]).astype(np.int32)

        data_dict = dict(
            coord=coord,
            strength=strength,
            segment=segment,
            name=self.get_data_name(idx),

        )
        data_dict["segment"][data_dict["segment"]==-1] =1
        print(data_dict["segment"])
        print("data_dict max()")

        print(data_dict["segment"].max())
        print("data_dict min()")
        print(data_dict["segment"].min())
        return data_dict

    def OLD_get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        with open(data_path, "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        strength = scan[:, -1].reshape([-1, 1])

        label_file = data_path.replace("velodyne", "labels").replace(".bin", ".label")
        if os.path.exists(label_file):
            with open(label_file, "rb") as a:
                raw_segment = np.fromfile(a, dtype=np.int32).reshape(-1)
                # --- ADD THESE PRINT STATEMENTS ---
                print(f"Processing {data_path}")
                print(f"Raw segment unique values: {np.unique(raw_segment)}")
                print(f"Raw segment min: {raw_segment.min()}, max: {raw_segment.max()}")
                # --- END ADDED PRINT STATEMENTS ---
                segment = np.vectorize(self.learning_map.__getitem__)(
                    raw_segment & 0xFFFF
                ).astype(np.int32)
                # --- ADD THESE PRINT STATEMENTS ---
                print(f"Mapped segment unique values: {np.unique(segment)}")
                print(f"Mapped segment min: {segment.min()}, max: {segment.max()}")
                # --- END ADDED PRINT STATEMENTS ---
        else:
            segment = np.zeros(scan.shape[0]).astype(np.int32)

        data_dict = dict(
            coord=coord,
            strength=strength,
            segment=segment,
            name=self.get_data_name(idx),
        )
        print("coord:"+str(coord.shape))
        print("segment:"+str(segment.shape))
        #print("name:"+str(data_dict["name"].shape))
        return data_dict

    def old_get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        with open(data_path, "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        strength = scan[:, -1].reshape([-1, 1])

        label_file = data_path.replace("velodyne", "labels").replace(".bin", ".label")
        if os.path.exists(label_file):
            with open(label_file, "rb") as a:
                segment = np.fromfile(a, dtype=np.int32).reshape(-1)
                segment = np.vectorize(self.learning_map.__getitem__)(
                    segment & 0xFFFF
                ).astype(np.int32)
        else:
            segment = np.zeros(scan.shape[0]).astype(np.int32)




        data_dict = dict(
            coord=coord,
            strength=strength,
            segment=segment,
            name=self.get_data_name(idx),
        )
        return data_dict

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        dir_path, file_name = os.path.split(file_path)
        sequence_name = os.path.basename(os.path.dirname(dir_path))
        frame_name = os.path.splitext(file_name)[0]
        data_name = f"{sequence_name}_{frame_name}"
        return data_name

    @staticmethod
    def get_learning_map(ignore_index):
        # Your specific dataset labels mapped to 0-13 for 14 classes


        #orignal nmubers 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 17, 18, 64, 66, 67, 68

        return {
            -1: 1,  # Ground
            1: 0,  # Ground
            2: 1,  # Railway
            3: 2,  # Roads
            4: 3,  # Veg.Low
            5: 4,  # Veg.Medium
            6: 5,  # Veg.High
            7: ignore_index,
            9: 6,  # Buildings
            10: 7, # Water
            11: 8, # Bridges
            12: 9, # Vehicles
            14:ignore_index,
            15: 10, # Electrical_Towers
            17: 11, # Power_lines
            18: 12, # Solar_panels
            64: 13, # Wind_turbines
            # Map other unwanted labels to ignore_index
            66: ignore_index,
            67: ignore_index,
            68: ignore_index,
            # Add any other labels present in your raw data that should be ignored or mapped
            255: ignore_index  # Common for unknown/unlabeled in some datasets
        }

    @staticmethod
    def get_learning_map_inv(ignore_index):
        # Inverse mapping from 0-13 back to your original dataset labels
        return {
            ignore_index: 255, # Assuming 255 was the original ignore
            0: 1,   # Ground
            1: 2,   # Railway
            2: 3,   # Roads
            3: 4,   # Veg.Low
            4: 5,   # Veg.Medium
            5: 6,   # Veg.High
            6: 9,   # Buildings
            7: 10,  # Water
            8: 11,  # Bridges
            9: 12,  # Vehicles
            10: 15, # Electrical_Towers
            11: 17, # Power_lines
            12: 18, # Solar_panels
            13: 64, # Wind_turbines
            # Note: 66 and 67 were mapped to ignore_index, so they don't have an inverse mapping here unless you specifically want to map ignore_index to one of them.
        }
