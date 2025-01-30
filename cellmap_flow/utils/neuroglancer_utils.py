import neuroglancer
import itertools
import logging

neuroglancer.set_server_bind_address("0.0.0.0")

logger = logging.getLogger(__name__)

from cellmap_flow.image_data_interface import ImageDataInterface
# TODO support multiresolution datasets
def get_raw_layer(dataset_path, filetype):
    if filetype == "zarr":
        axis = ["x", "y", "z"]
    else:
        axis = ["z", "y", "x"]
    image = ImageDataInterface(dataset_path)
    return neuroglancer.ImageLayer(
        source=neuroglancer.LocalVolume(
            data=image.ts,
            dimensions=neuroglancer.CoordinateSpace(
                names=axis,
                units="nm",
                scales=image.voxel_size,
            ),
            voxel_offset=image.offset,
            )
        )

from funlib.geometry.coordinate import Coordinate
from cellmap_flow.image_data_interface import ImageDataInterface
from cellmap_flow.inferencer import Inferencer
from cellmap_flow.utils.data import ModelConfig
from cellmap_flow.local_inferencer import InferencerLocalVolume

def get_inference_layer(dataset_name: str, model_config: ModelConfig, filetype: str):
    input_voxel_size = Coordinate(model_config.config.input_voxel_size)
    # block_shape = [int(x) for x in model_config.config.block_shape]


    inferencer = Inferencer(model_config)

    # Load or initialize your dataset
    idi_raw = ImageDataInterface(
        dataset_name, target_resolution=input_voxel_size
    )
    
    if filetype == "zarr":
        axis = ["x", "y", "z", "c^"]
    else:
        axis = ["z", "y", "x","c^"]

    inference_volume = InferencerLocalVolume(
    inferencer=inferencer,
    idi_raw=idi_raw,
    block_size=(56, 56, 56),
    dimensions=neuroglancer.CoordinateSpace(
                names=axis,
                units=["nm", "nm", "nm", ""],
                scales=[8,8,8,1],
            ),
    volume_type="image",  # or "segmentation"
    voxel_offset=[0, 0, 0, 0], 
    encoding="npz",
    downsampling=None,
        )
    return neuroglancer.ImageLayer(
        source=inference_volume,
        )

def generate_local_neuroglancer_link(dataset_path, model_config):
    viewer = neuroglancer.UnsynchronizedViewer()
    with viewer.txn() as s:
        s.layers["raw"] = get_raw_layer(dataset_path, "zarr")
        s.layers["inference"] = get_inference_layer(dataset_path, model_config, "zarr")
        show(str(viewer))
        while True:
            pass

def generate_neuroglancer_link(dataset_path, inference_dict):
    # Create a new viewer
    viewer = neuroglancer.UnsynchronizedViewer()

    # Add a layer to the viewer
    with viewer.txn() as s:
        # if multiscale dataset
        # if (
        #     dataset_path.split("/")[-1].startswith("s")
        #     and dataset_path.split("/")[-1][1:].isdigit()
        # ):
        #     dataset_path = dataset_path.rsplit("/", 1)[0]
        if ".zarr" in dataset_path:
            filetype = "zarr"
        elif ".n5" in dataset_path:
            filetype = "n5"
        else:
            filetype = "precomputed"
        if dataset_path.startswith("/"):
            s.layers["raw"] = get_raw_layer(dataset_path, filetype)
            # if "nrs/cellmap" in dataset_path:
            #     security = "https"
            #     dataset_path = dataset_path.replace("/nrs/cellmap/", "nrs/")
            # elif "/groups/cellmap/cellmap" in dataset_path:
            #     security = "http"
            #     dataset_path = dataset_path.replace("/groups/cellmap/cellmap/", "dm11/")
            # else:
            #     raise ValueError(
            #         "Currently only supporting nrs/cellmap and /groups/cellmap/cellmap"
            #     )

            # s.layers["raw"] = neuroglancer.ImageLayer(
            #     source=f"{filetype}://{security}://cellmap-vm1.int.janelia.org/{dataset_path}",
            # )
        else:
            s.layers["raw"] = neuroglancer.ImageLayer(
                source=f"{filetype}://{dataset_path}",
            )
        colors = [
            "red",
            "green",
            "blue",
            "yellow",
            "purple",
            "orange",
            "cyan",
            "magenta",
        ]
        color_cycle = itertools.cycle(colors)
        for host, model in inference_dict.items():
            color = next(color_cycle)
            s.layers[model] = neuroglancer.ImageLayer(
                source=f"n5://{host}/{model}",
                shader=f"""#uicontrol invlerp normalized(range=[0, 255], window=[0, 255]);
#uicontrol vec3 color color(default="{color}");
void main(){{emitRGB(color * normalized());}}""",
            )
        # print(viewer)  # neuroglancer.to_url(viewer.state))
        show(str(viewer))
        # logger.error(f"\n \n \n link : {viewer}")
        while True:
            pass


def show(viewer):
    print()
    print()
    print("**********************************************")
    print("LINK:")
    print(viewer)
    print("**********************************************")
    print()
    print()
