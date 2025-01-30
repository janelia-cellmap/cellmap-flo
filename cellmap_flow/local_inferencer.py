import math
import numpy as np
from funlib.geometry import Roi

from neuroglancer.local_volume import LocalVolume
from neuroglancer.chunks import encode_jpeg, encode_npz, encode_raw

class InferencerLocalVolume(LocalVolume):
    """
    A LocalVolume that forces requests for a particular 3D block size (e.g. 32x32x32)
    for the model input, and returns a 4D output (z,y,x,c).
    """

    def __init__(
        self,
        inferencer,
        idi_raw,
        dimensions,
        block_size=(32, 32, 32),
        voxel_offset=None,
        volume_type="image",
        encoding="npz",
        downsampling=None,
        chunk_layout=None,
        max_downsampling=1,
        max_downsampled_size=math.inf,
        max_downsampling_scales=1,
        max_voxels_per_chunk_log2=None,
    ):
        """
        :param inferencer: Your model wrapper with a `process_chunk(idi, roi)` method
                           that returns a 4D result [z, y, x, channels].
        :param idi_raw: Handle to the raw data for reading inputs (3D).
        :param dimensions: Neuroglancer CoordinateSpace with the shape of the final 4D volume.
        :param block_size: The 3D input block size for your model, e.g. (32, 32, 32).
        :param volume_type: "image" or "segmentation".
        :param encoding: "npz", "raw", "jpeg", etc.
        :param downsampling: Typically None if you only want the native resolution.
        ...
        """
        # For demonstration, suppose idi_raw.shape is (Z, Y, X).
        # The model *output* adds channels, e.g. shape -> (Z, Y, X, C).

        raw_shape_3d = idi_raw.shape  # e.g. (Z, Y, X)
        output_channels = getattr(
            inferencer.model_config.config, "output_channels", 1
        )
        # The final volume shape as seen by Neuroglancer is 4D:
        shape_4d = (raw_shape_3d[0], raw_shape_3d[1], raw_shape_3d[2], output_channels)

        self.inferencer = inferencer
        self.idi_raw = idi_raw
        self.block_shape = block_size
        self.block_size = block_size

        # Create a minimal data object so LocalVolume can store shape/dtype:
        class _FakeData:
            def __init__(self, shape, dtype, rank):
                self.shape = shape
                self.dtype = dtype
                self.rank = rank

            def __getitem__(self, key):
                # Not actually used, but must exist
                return np.zeros((1,), dtype=self.dtype)

        fake_data = _FakeData(shape=shape_4d, dtype=np.uint8, rank=len(shape_4d))

        super().__init__(
            data=fake_data,
            dimensions=dimensions,
            volume_type=volume_type,
            voxel_offset=voxel_offset,
            encoding=encoding,
            downsampling=downsampling,
            chunk_layout=chunk_layout,
            max_downsampling=max_downsampling,
            max_downsampled_size=max_downsampled_size,
            max_downsampling_scales=max_downsampling_scales,
            max_voxels_per_chunk_log2=max_voxels_per_chunk_log2,
        )


        

    def get_encoded_subvolume(self, data_format, start, end, scale_key):
        """
        Example usage: we chunk the request, gather data, then encode.
        """
        # Parse the request: e.g. start=[64,8192,16896,0], end=[96,8224,16928,8]
        # We'll assume only scale=1,1,1,1 is supported, or parse it if you like.
        
        # 1) Use the helper function
        subvol = self.get_chunked_subvolume(
            start, end,
            block_size=self.block_size,     # your desired block size, e.g. (32,32,32)
        )

        # 2) Encode for Neuroglancer
        if data_format == "npz":
            from neuroglancer.chunks import encode_npz
            data = encode_npz(subvol)
            content_type = "application/octet-stream"
        elif data_format == "raw":
            from neuroglancer.chunks import encode_raw
            data = encode_raw(subvol)
            content_type = "application/octet-stream"
        elif data_format == "jpeg":
            from neuroglancer.chunks import encode_jpeg
            data = encode_jpeg(subvol)
            content_type = "image/jpeg"
        else:
            raise ValueError(f"Unsupported data format: {data_format}")

        return data, content_type


    def _read_model_chunk(
        self,
        global_z0, global_y0, global_x0,
        bz, by, bx,
        channel_start, channel_end
    ):
        """
        Example chunk-reading or model inference. We form an ROI and call our model.
        Suppose self.inferencer.process_chunk(roi) returns shape (bz, by, bx, c).
        """
        from funlib.geometry import Roi, Coordinate

        # Build the bounding box for the chunk in global coords
        start_3d = (global_z0, global_y0, global_x0)
        size_3d = (bz, by, bx)
        roi_3d = Roi(start_3d, size_3d)

        # Run the model or read the chunk from an array
        chunk_data_4d = self.inferencer.process_chunk(self.idi_raw, roi_3d)
        # chunk_data_4d => shape (bz, by, bx, C)

        # Possibly slice the channel dimension if you want partial channels
        return chunk_data_4d[..., channel_start:channel_end]
    
    def read_model_chunk_func(self,chunk_x, chunk_y, chunk_z,
                          bz, by, bx,
                          channel_start, channel_end):
        corner = self.block_shape[:3] * np.array([chunk_z, chunk_y, chunk_x])
        box = np.array([corner, self.block_shape[:3]]) * np.array([8,8,8])
        roi = Roi(box[0], box[1])
        print(f"Requesting chunk {roi} - ts: {self.idi_raw.ts.shape}")
        chunk_data = self.inferencer.process_chunk(self.idi_raw, roi)
        return chunk_data
        


    def get_chunked_subvolume(self,
        start, end,            # [start_z, start_y, start_x, start_c], [end_z, end_y, end_x, end_c]
        block_size,            # (block_size_z, block_size_y, block_size_x)
    ):
        """
        Retrieve a subvolume [start, end) by splitting into block-sized chunks
        in z,y,x. We then merge them into a single array of shape 
        (end_z - start_z, end_y - start_y, end_x - start_x, end_c - start_c).

        :param start: 4D start coordinates, e.g. (64, 8192, 16896, 0)
        :param end:   4D end coordinates,   e.g. (96, 8224, 16928,  8)
        :param block_size: (bz, by, bx), e.g. (32, 32, 32)
        :param read_model_chunk_func: a callback that takes chunk indices or bounding box
                                    and returns a (bz, by, bx, c)-shaped np.ndarray.
        :return: np.ndarray of shape (sz, sy, sx, sc) 
                where sz=end_z-start_z, etc.
        """
        s_z, s_y, s_x, s_c = start
        e_z, e_y, e_x, e_c = end
        bz, by, bx = block_size

        # The shape of the final subvolume to return
        out_shape = (e_z - s_z, e_y - s_y, e_x - s_x, e_c - s_c)
        out = np.zeros(out_shape, dtype=np.uint8)

        # We only chunk in z,y,x. For channels, we typically read the entire range at once,
        # or if you want to chunk channels too, you'd do the same approach.
        channel_start = s_c
        channel_end   = e_c

        # 1) Determine chunk index ranges in each spatial dimension
        z_chunk_start = s_z // bz
        z_chunk_end   = (e_z - 1) // bz
        y_chunk_start = s_y // by
        y_chunk_end   = (e_y - 1) // by
        x_chunk_start = s_x // bx
        x_chunk_end   = (e_x - 1) // bx

        # 2) Loop over chunk indices
        for zc in range(z_chunk_start, z_chunk_end + 1):
            chunk_z0 = zc * bz
            chunk_z1 = chunk_z0 + bz

            # Intersection with the requested [s_z, e_z)
            # We only really need the portion that overlaps the request
            read_z0 = max(chunk_z0, s_z)
            read_z1 = min(chunk_z1, e_z)

            for yc in range(y_chunk_start, y_chunk_end + 1):
                chunk_y0 = yc * by
                chunk_y1 = chunk_y0 + by
                read_y0 = max(chunk_y0, s_y)
                read_y1 = min(chunk_y1, e_y)

                for xc in range(x_chunk_start, x_chunk_end + 1):
                    chunk_x0 = xc * bx
                    chunk_x1 = chunk_x0 + bx
                    read_x0 = max(chunk_x0, s_x)
                    read_x1 = min(chunk_x1, e_x)

                    # The chunk bounding box in global coords is [chunk_*0, chunk_*1).
                    # We'll read the entire chunk from the model (which might produce (bz,by,bx,c)).
                    # Then we slice out the part that overlaps our requested [start..end).

                    model_chunk_data = self.read_model_chunk_func(
                        chunk_z0, chunk_y0, chunk_x0,  # chunk's top-left in global space
                        bz, by, bx,                   # chunk shape in z,y,x
                        channel_start, channel_end    # channel range
                    )
                    # "model_chunk_data" is typically shape (bz, by, bx, c)

                    # Figure out how to place it into "out".
                    # The offsets into the final output are (read_* - s_*) in each dimension.
                    out_z0 = read_z0 - s_z
                    out_z1 = read_z1 - s_z
                    out_y0 = read_y0 - s_y
                    out_y1 = read_y1 - s_y
                    out_x0 = read_x0 - s_x
                    out_x1 = read_x1 - s_x

                    # The offsets within the chunk data are (read_*0 - chunk_*0).
                    in_z0 = read_z0 - chunk_z0
                    in_z1 = read_z1 - chunk_z0
                    in_y0 = read_y0 - chunk_y0
                    in_y1 = read_y1 - chunk_y0
                    in_x0 = read_x0 - chunk_x0
                    in_x1 = read_x1 - chunk_x0

                    # Place that overlap region into "out"
                    out[out_z0:out_z1, out_y0:out_y1, out_x0:out_x1, :] = \
                        model_chunk_data[in_z0:in_z1, in_y0:in_y1, in_x0:in_x1, :]

        return out
