"""Tests for field handling with VoxelFields class."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import evoxels as evo


def test_voxelFields_init():
    N = 10
    vf = evo.VoxelFields((N, N, N))
    assert vf.shape == (N, N, N)
    assert (vf.Nx, vf.Ny, vf.Nz) == (N, N, N)

def test_voxelFields_init_domain():
    N = 10
    vf = evo.VoxelFields((N, N, N), (N, N, N))
    assert (vf.domain_size, vf.spacing) == ((N, N, N), (1, 1, 1))

def test_voxelFields_grid_cell_centered():
    Nx, Ny, Nz = 10, 5, 7
    vf = evo.VoxelFields((Nx, Ny, Nz), (Nx, Ny, Nz))
    x, _, _ = vf.meshgrid()
    assert x[-1,0,0] == Nx-vf.spacing[0]/2

def test_voxelFields_grid_staggered_x():
    Nx, Ny, Nz = 10, 5, 7
    vf = evo.VoxelFields((Nx + 1, Ny, Nz), (Nx, Ny, Nz), convention='staggered_x')
    x, y, _ = vf.meshgrid()
    assert (x[-1,0,0] == Nx) and (y[0,-1,0] == Ny-vf.spacing[1]/2)

def test_voxelFields_init_fields():
    N = 10
    vf = evo.VoxelFields((N, N, N), (N, N, N))
    vf.add_field("c", 0.123*np.ones((N, N, N)))

    assert (vf.fields['c'][1,2,3], *vf.fields['c'].shape) == (0.123, N, N, N)

@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive:UserWarning")
def test_plot_functions():
    vf = evo.VoxelFields((3, 3, 3))
    vf.add_field("f", np.arange(27).reshape(vf.shape))
    with pytest.raises(ValueError):
        vf.plot_slice("f", 0, direction="bad")
    vf.plot_slice("f", 0, direction="x")
    vf.plot_slice("f", 1, direction="y")
    vf.plot_slice("f", 2, direction="z")
    slider = vf.plot_field_interactive("f", direction="y")
    slider.set_val(1)
    plt.close("all")


@pytest.mark.parametrize("direction, axis", [("x", 0), ("y", 1), ("z", 2)])
def test_plot_background(direction, axis, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    vf = evo.VoxelFields((3, 4, 5))
    values = np.arange(60).reshape(vf.shape)
    background = values % 3 == 0
    values[background] = 10000
    vf.add_field("f", values)
    original_bad = plt.get_cmap("viridis").get_bad().copy()

    assert vf.plot_slice("f", 1, direction=direction, background=background,
                         background_color="red") is None
    im = plt.gcf().axes[0].images[0]
    expected_mask = np.take(background, 1, axis=axis).T
    np.testing.assert_array_equal(im.get_array().mask, expected_mask)
    visible = np.take(values, 1, axis=axis).T[~expected_mask]
    assert im.get_clim() == (visible.min(), visible.max())
    np.testing.assert_array_equal(im.cmap.get_bad(), [1, 0, 0, 1])

    slider = vf.plot_field_interactive("f", direction=direction, background=background)
    im = slider.ax.figure.axes[0].images[0]
    for index in (0, 1, 2):
        slider.set_val(index)
        np.testing.assert_array_equal(
            im.get_array().mask, np.take(background, index, axis=axis).T,
        )
    assert im.get_clim() == (values[~background].min(), values[~background].max())
    np.testing.assert_array_equal(im.cmap.get_bad(), [0, 0, 0, 1])
    np.testing.assert_array_equal(plt.get_cmap("viridis").get_bad(), original_bad)
    np.testing.assert_array_equal(vf.fields["f"], values)
    plt.close("all")


@pytest.mark.parametrize("interactive", [False, True])
def test_plot_background_validation_and_empty(interactive, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    vf = evo.VoxelFields((3, 4, 5))
    vf.add_field("f", np.ones(vf.shape))
    plot = vf.plot_field_interactive if interactive else lambda *a, **kw: vf.plot_slice(*a, 0, **kw)
    with pytest.raises(ValueError, match="boolean"):
        plot("f", background=np.zeros(vf.shape, dtype=int))
    with pytest.raises(ValueError, match="shape"):
        plot("f", background=np.zeros((3, 4), dtype=bool))
    for bounds in (None, (2, 5)):
        plot("f", background=np.ones(vf.shape, dtype=bool), value_bounds=bounds)
        im = plt.gcf().axes[0].images[0]
        assert im.get_array().mask.all()
        assert im.get_clim() == ((0, 1) if bounds is None else bounds)
    plt.close("all")

def test_mock_vtk_export():
    vf = evo.VoxelFields((3, 3, 3))
    vf.add_field('test1')
    vf.add_field('test2')
    with pytest.raises(ValueError):
        vf.export_to_vtk(filename='bad_name')

def test_npy_export(tmp_path):
    vf = evo.VoxelFields((2, 3, 1))
    field = np.arange(6).reshape(vf.shape)
    vf.add_field("test", field)

    filename = tmp_path / "field.npy"
    vf.export_to_npy("test", filename)

    assert np.array_equal(np.load(filename), field)

def test_sphere_volume_fraction():
    vf = evo.VoxelFields((6, 5, 5), convention='staggered_x')
    vf.add_field("sphere")
    vf.set_voxel_sphere("sphere", center=(0.5, 0.5, 0.5), radius=0.31, label=1)
    assert np.count_nonzero(vf.fields["sphere"] == 1) == 20
    assert vf.average('sphere') == 0.16
