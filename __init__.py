"""
Annotate 3d polylines with CVAT.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import numpy as np

import fiftyone as fo
import fiftyone.core.storage as fos
import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone.utils.utils3d as fou3d
import fiftyone.zoo as foz


_PROJECTION_SLICE = "projection"
_PROJECTION_FIELD = "polyline_2d"


def create_projection_slice_if_necessary(ctx, pcd_slice):
    if _PROJECTION_SLICE not in ctx.dataset.group_slices:
        ## Create opm
        output_dir = fos.make_temp_dir()
        fou3d.compute_orthographic_projection_images(
            ctx.dataset,
            output_dir=output_dir,
            size=(-1, 1080),
            in_group_slice=pcd_slice,
            out_group_slice=_PROJECTION_SLICE,
            shading_mode="intensity",
        )


def get_anno_tag_view(ctx):
    view = ctx.dataset
    if "annotate" in ctx.dataset.select_group_slices(_allow_mixed=True).distinct("tags"):
        gids = ctx.dataset.select_group_slices(_allow_mixed=True).match_tags("annotate").distinct("group.id")
        view = ctx.dataset.select_groups(gids)
    return view

def get_view(ctx):
    view = get_anno_tag_view(ctx)
    view = view.select_group_slices(_PROJECTION_SLICE)
    return view


def request_annotation(
        view,
        anno_key,
        classes,
    ):
    ## Upload to CVAT to annotate polylines
    results = view.annotate(anno_key, label_field=_PROJECTION_FIELD, label_type="polylines", classes=classes)


def polyline_2d_to_3d(polyline_2d, metadata, road_z_value):
    min_bound = metadata.min_bound
    max_bound = metadata.max_bound
    width = metadata.width
    height = metadata.height
    shape = np.array(polyline_2d.points).shape
    points = np.zeros((shape[0], shape[1], shape[2]+1))
    points[:,:,-1] = road_z_value
    points[:,:,:-1] = np.array(polyline_2d.points)
    points[:,:, 1] *= -1
    points[:,:, 1] += 1
    points[:,:, 0] *= (max_bound[0] - min_bound[0])
    points[:,:, 1] *= (max_bound[1] - min_bound[1])
    points[:,:, 0] += min_bound[0]
    points[:,:, 1] += min_bound[1]

    return fo.Polyline(points3d=points.tolist(), label=polyline_2d.label)


def load_annotation(
        dataset,
        anno_key,
        label_field,
        pcd_slice,
        road_surface,
    ):

    view = dataset.load_annotation_view(anno_key)
    dataset.load_annotations(anno_key)
    results = view.load_annotation_results(anno_key)

    # Transform polylines to pcd
    all_polylines, metadatas, group_ids = view.values([_PROJECTION_FIELD, "orthographic_projection_metadata", "group.id"])

    polylines_3d = {}
    for sample_polylines, metadata, group_id in zip(all_polylines, metadatas, group_ids):
        sample_polylines_3d = []
        if sample_polylines is not None:
            for p in sample_polylines.polylines:
                polyline_3d = polyline_2d_to_3d(p, metadata, road_surface)
                sample_polylines_3d.append(polyline_3d)

        polylines_3d[group_id] = fo.Polylines(polylines=sample_polylines_3d)

    view._dataset.select_group_slices(pcd_slice).set_values(label_field, polylines_3d, key_field="group.id")


class Annotate3DPolyline(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="annotate_3d_polyline",
            label="Annotate 3d polyline",
            dynamic=True,
        )
    def __call__(
        self,
        samples,
        anno_key,
        classes,
        pcd_slice,
    ):
        ctx = dict(view=samples.view(), dataset=samples.view()._dataset)
        params = dict(
            anno_key=anno_key,
            classes=classes,
            pcd_slice=pcd_slice,
        )
        return foo.execute_operator(self.uri, ctx, params=params)

    def resolve_input(self, ctx):
        inputs = types.Object()

        inputs.str(
            "anno_key",
            required=True,
            label="Annotation key",
            description=(
                "Unique name for this annotation run"
            ),
        )

        slice_choices = types.DropdownView(space=6)
        default_pcd_slice = None
        for s, t in ctx.dataset.group_media_types.items():
            if t in ["pcd", "3d"]:
                slice_choices.add_choice(s, label=s)
                if default_pcd_slice is None:
                    default_pcd_slice = s
        inputs.str(
            "pcd_slice",
            required=True,
            label="PCD Slice",
            description="The slice with the pcd to annotate",
            view=slice_choices,
            default=default_pcd_slice,
        )

        inputs.list(
            "classes",
            types.String(),
            label="Classes",
            description="The classes to annotate",
            required=True,
        )

        return types.Property(inputs, view=types.View(label="Annotate"))

    def execute(self, ctx):
        anno_key = ctx.params.get("anno_key", None)
        classes = ctx.params.get("classes", [])
        pcd_slice = ctx.params.get("pcd_slice", None)
        create_projection_slice_if_necessary(ctx, pcd_slice)
        view = get_view(ctx)

        request_annotation(
            view,
            anno_key,
            classes,
        )


class Load3DPolyline(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="load_3d_polyline",
            label="Load 3d polyline",
            dynamic=True,
        )

    def __call__(
        self,
        samples,
        anno_key,
        label_field,
        pcd_slice,
        road_surface=-1.6,
    ):
        ctx = dict(view=samples.view(), dataset=samples.view()._dataset)
        params = dict(
            anno_key=anno_key,
            pcd_slice=pcd_slice,
            label_field=label_field,
            road_surface=road_surface,
        )
        return foo.execute_operator(self.uri, ctx, params=params)

    def resolve_input(self, ctx):
        inputs = types.Object()

        anno_keys = types.DropdownView(space=6)
        for anno_key in ctx.dataset.list_annotation_runs():
            anno_keys.add_choice(anno_key, label=anno_key)

        inputs.str(
            "anno_key",
            required=True,
            label="Annotation key",
            description=(
                "Unique name for this annotation run"
            ),
            view=anno_keys,
        )

        slice_choices = types.DropdownView(space=6)
        default_pcd_slice = None
        for s, t in ctx.dataset.group_media_types.items():
            if t in ["pcd", "3d"]:
                slice_choices.add_choice(s, label=s)
                if default_pcd_slice is None:
                    default_pcd_slice = s


        inputs.str(
            "pcd_slice",
            required=True,
            label="PCD Slice",
            description="The slice with the pcd to annotate",
            view=slice_choices,
            default=default_pcd_slice,
        )

        inputs.str(
            "label_field",
            required=True,
            label="Label field name",
            description=(
                "The name for the label field to store the 3d annotations"
            ),
        )

        inputs.float(
            "road_surface",
            default=-1.6,
            label="Road height",
            description=(
                "The height of the road surface in the Z direction in the PCD"
            ),
            required=True,
        )

        return types.Property(inputs, view=types.View(label="Annotate"))

    def execute(self, ctx):
        anno_key = ctx.params.get("anno_key", None)
        label_field = ctx.params.get("label_field", None)
        pcd_slice = ctx.params.get("pcd_slice", None)
        road_surface = ctx.params.get("road_surface", 0)

        load_annotation(
            ctx.dataset,
            anno_key,
            label_field,
            pcd_slice,
            road_surface,
        )
        ctx.trigger("reload_dataset")


def register(p):
    p.register(Annotate3DPolyline)
    p.register(Load3DPolyline)
