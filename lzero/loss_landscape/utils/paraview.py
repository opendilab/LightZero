"""
Overview:
    ParaView VTP export utilities for professional-quality loss landscape visualization.
    This module enables export of loss surfaces to VTP (VTK XML PolyData) format,
    allowing interactive 3D exploration in ParaView with advanced rendering capabilities.

This module provides:
    - HDF5 to VTP conversion with customizable resolution and scaling
    - Surface interpolation for smooth high-resolution rendering
    - Logarithmic scaling support for large loss value ranges
    - Z-axis clipping to focus on regions of interest
    - Point data and cell data export for flexible ParaView workflows

Key Functions:
    - h5_to_vtp: Convert HDF5 loss surface to ParaView-compatible VTP format
    - _write_vtp_file: Internal VTK XML writer (private helper function)

Notes:
    - ParaView advantages: Professional rendering, lighting effects, high-DPI export
    - VTP files preserve both point-wise and cell-averaged values
    - Interpolation useful for smooth surfaces from coarse grid computations
    - Log scaling recommended for loss landscapes spanning multiple orders of magnitude
"""

import math
from typing import Optional, List
import h5py
import numpy as np
from scipy import interpolate


def h5_to_vtp(
    surf_file: str,
    surf_name: str = 'train_loss',
    log: bool = False,
    zmax: float = -1,
    interp: int = -1,
    output: Optional[str] = None
) -> None:
    """
    Overview:
        Convert HDF5 loss surface to VTP (VTK XML PolyData) format for ParaView visualization.
        Enables professional 3D rendering with advanced lighting, shading, and export capabilities
        not available in standard matplotlib plots.

    Arguments:
        - surf_file (:obj:`str`): Path to HDF5 surface file containing loss landscape data
        - surf_name (:obj:`str`, optional): Name of surface dataset to export. Default is 'train_loss'
        - log (:obj:`bool`, optional): Apply logarithmic scale to z values (log(z + 0.1)). Default is False
        - zmax (:obj:`float`, optional): Maximum z value for clipping. Values above are set to zmax.
            Use -1 to disable clipping. Default is -1
        - interp (:obj:`int`, optional): Interpolate surface to NxN resolution using cubic interpolation.
            Use -1 to disable interpolation. Default is -1
        - output (:obj:`str`, optional): Output VTP file path. If None, auto-generates from surf_file
            with format: {surf_file}_{surf_name}[_zmax={zmax}][_log].vtp. Default is None

    Returns:
        - None: Writes VTP file to disk at the specified or auto-generated path

    Notes:
        - VTP format includes both point data (per-vertex z values) and cell data (averaged per-polygon)
        - Logarithmic scaling useful for loss ranges spanning orders of magnitude (e.g., 0.01 to 100)
        - Interpolation creates smoother surfaces but may introduce artifacts if grid is too coarse
        - Clipping with zmax helps focus on interesting regions when outliers dominate the scale
        - ParaView recommended settings: Apply 'Surface with Edges', enable lighting for depth perception

    Examples::
        >>> # Basic export for ParaView visualization
        >>> h5_to_vtp('model_surface.h5', surf_name='train_loss')

        >>> # Export with log scale and clipping for wide loss range
        >>> h5_to_vtp('model_surface.h5', log=True, zmax=10, surf_name='train_loss')

        >>> # High-resolution export with interpolation
        >>> h5_to_vtp('model_surface.h5', interp=200, output='smooth_landscape.vtp')
    """
    print('-' * 60)
    print('Converting HDF5 to VTP format for ParaView')
    print('-' * 60)

    # Load surface data from HDF5 file
    f = h5py.File(surf_file, 'r')

    [xcoordinates, ycoordinates] = np.meshgrid(f['xcoordinates'][:], f['ycoordinates'][:])
    vals = f[surf_name][:]

    # Flatten 2D grids to 1D arrays for VTP format
    x_array = xcoordinates[:].ravel()
    y_array = ycoordinates[:].ravel()
    z_array = vals.ravel()

    # Apply cubic interpolation to create smoother surface if requested
    if interp > 0:
        print(f"Interpolating surface to {interp}x{interp} resolution...")
        m = interpolate.interp2d(xcoordinates[0, :], ycoordinates[:, 0], vals, kind='cubic')
        x_new = np.linspace(min(x_array), max(x_array), interp)
        y_new = np.linspace(min(y_array), max(y_array), interp)
        z_new = m(x_new, y_new).ravel()

        x_new, y_new = np.meshgrid(x_new, y_new)
        x_array = x_new.ravel()
        y_array = y_new.ravel()
        z_array = z_new

    # Auto-generate output filename with descriptive suffixes if not provided
    if output is None:
        output = surf_file + f"_{surf_name}"
        if zmax > 0:
            output += f"_zmax={zmax}"
        if log:
            output += "_log"
        output += ".vtp"

    # Clip extreme z values to focus on region of interest
    if zmax > 0:
        z_array[z_array > zmax] = zmax

    # Apply logarithmic transformation to handle wide value ranges
    if log:
        z_array = np.log(z_array + 0.1)  # Add offset to avoid log(0)

    print(f"Output file: {output}")

    # Calculate mesh geometry parameters for structured grid
    number_points = len(z_array)
    matrix_size = int(math.sqrt(number_points))  # Assumes square grid
    poly_size = matrix_size - 1  # Number of cells is one less than points
    number_polys = poly_size * poly_size

    print(f"Number of points: {number_points}")
    print(f"Matrix size: {matrix_size} x {matrix_size}")
    print(f"Number of polygons: {number_polys}")

    # Compute value ranges for VTP metadata
    min_val = min(min(x_array), min(y_array), min(z_array))
    max_val = max(max(x_array), max(y_array), max(z_array))

    # Compute cell-centered z values by averaging neighboring vertices
    averaged_z_values = []
    for col in range(poly_size):
        stride = col * matrix_size
        for row in range(poly_size):
            idx = stride + row
            avg_z = (z_array[idx] + z_array[idx + 1] +
                     z_array[idx + matrix_size] + z_array[idx + matrix_size + 1]) / 4.0
            averaged_z_values.append(avg_z)

    # Generate VTP file in VTK XML format
    _write_vtp_file(output, x_array, y_array, z_array, averaged_z_values,
                    number_points, matrix_size, poly_size, number_polys,
                    min_val, max_val)

    f.close()
    print(f"Successfully exported to VTP format!")
    print(f"Open '{output}' in ParaView for interactive 3D visualization.")


def _write_vtp_file(
    vtp_file: str,
    x_array: np.ndarray,
    y_array: np.ndarray,
    z_array: np.ndarray,
    averaged_z_values: List[float],
    number_points: int,
    matrix_size: int,
    poly_size: int,
    number_polys: int,
    min_val: float,
    max_val: float
) -> None:
    """
    Overview:
        Internal helper function to write VTP (VTK XML PolyData) file for ParaView.
        Generates well-formed XML with point data, cell data, coordinates, and polygon connectivity.

    Arguments:
        - vtp_file (:obj:`str`): Output file path for VTP file
        - x_array (:obj:`numpy.ndarray`): Flattened x-coordinates of all points
        - y_array (:obj:`numpy.ndarray`): Flattened y-coordinates of all points
        - z_array (:obj:`numpy.ndarray`): Flattened z-coordinates (loss values) of all points
        - averaged_z_values (:obj:`list`): Cell-averaged z values for polygon centers
        - number_points (:obj:`int`): Total number of vertices in the mesh
        - matrix_size (:obj:`int`): Size of the square grid (points per side)
        - poly_size (:obj:`int`): Number of polygons per side (matrix_size - 1)
        - number_polys (:obj:`int`): Total number of quadrilateral polygons
        - min_val (:obj:`float`): Minimum coordinate value across all dimensions
        - max_val (:obj:`float`): Maximum coordinate value across all dimensions

    Returns:
        - None: Writes VTP file to disk

    Notes:
        - Private helper function, not intended for direct external use
        - VTP format stores: PointData (vertex attributes), CellData (polygon attributes),
          Points (3D coordinates), and Polys (connectivity)
        - Uses quadrilateral (4-vertex) polygons for structured grid representation
        - ASCII format for readability and compatibility (binary format could be added)
    """
    with open(vtp_file, 'w') as f:
        # Write VTK XML header
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n')
        f.write('  <PolyData>\n')
        f.write(f'    <Piece NumberOfPoints="{number_points}" NumberOfVerts="0" NumberOfLines="0" '
                f'NumberOfStrips="0" NumberOfPolys="{number_polys}">\n')

        # Write point data (per-vertex attributes: loss values)
        f.write('      <PointData>\n')
        f.write(f'        <DataArray type="Float32" Name="zvalue" NumberOfComponents="1" '
                f'format="ascii" RangeMin="{min(z_array)}" RangeMax="{max(z_array)}">\n')
        for i, z in enumerate(z_array):
            if i % 6 == 0:
                f.write('          ')
            f.write(f'{z:.6f}')
            if i % 6 == 5:
                f.write('\n')
            else:
                f.write(' ')
        if len(z_array) % 6 != 0:
            f.write('\n')
        f.write('        </DataArray>\n')
        f.write('      </PointData>\n')

        # Cell Data (averaged z values)
        f.write('      <CellData>\n')
        f.write(f'        <DataArray type="Float32" Name="averaged_zvalue" NumberOfComponents="1" '
                f'format="ascii" RangeMin="{min(averaged_z_values)}" RangeMax="{max(averaged_z_values)}">\n')
        for i, z in enumerate(averaged_z_values):
            if i % 6 == 0:
                f.write('          ')
            f.write(f'{z:.6f}')
            if i % 6 == 5:
                f.write('\n')
            else:
                f.write(' ')
        if len(averaged_z_values) % 6 != 0:
            f.write('\n')
        f.write('        </DataArray>\n')
        f.write('      </CellData>\n')

        # Points (3D coordinates)
        f.write('      <Points>\n')
        f.write(f'        <DataArray type="Float32" Name="Points" NumberOfComponents="3" '
                f'format="ascii" RangeMin="{min_val}" RangeMax="{max_val}">\n')
        for i in range(number_points):
            if i % 2 == 0:
                f.write('          ')
            f.write(f'{x_array[i]:.6f} {y_array[i]:.6f} {z_array[i]:.6f}')
            if i % 2 == 1:
                f.write('\n')
            else:
                f.write(' ')
        if number_points % 2 != 0:
            f.write('\n')
        f.write('        </DataArray>\n')
        f.write('      </Points>\n')

        # Vertices (empty)
        f.write('      <Verts>\n')
        f.write('        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="0">\n')
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="0" RangeMax="0">\n')
        f.write('        </DataArray>\n')
        f.write('      </Verts>\n')

        # Lines (empty)
        f.write('      <Lines>\n')
        f.write('        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="0">\n')
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="0" RangeMax="0">\n')
        f.write('        </DataArray>\n')
        f.write('      </Lines>\n')

        # Strips (empty)
        f.write('      <Strips>\n')
        f.write('        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="0">\n')
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="0" RangeMax="0">\n')
        f.write('        </DataArray>\n')
        f.write('      </Strips>\n')

        # Polygons (surfaces)
        f.write('      <Polys>\n')
        f.write(f'        <DataArray type="Int64" Name="connectivity" format="ascii" '
                f'RangeMin="0" RangeMax="{number_points - 1}">\n')

        poly_count = 0
        for col in range(poly_size):
            stride = col * matrix_size
            for row in range(poly_size):
                idx = stride + row
                if poly_count % 2 == 0:
                    f.write('          ')
                f.write(f'{idx} {idx + 1} {idx + matrix_size + 1} {idx + matrix_size}')
                if poly_count % 2 == 1:
                    f.write('\n')
                else:
                    f.write(' ')
                poly_count += 1
        if poly_count % 2 == 1:
            f.write('\n')

        f.write('        </DataArray>\n')
        f.write(f'        <DataArray type="Int64" Name="offsets" format="ascii" '
                f'RangeMin="4" RangeMax="{number_polys * 4}">\n')

        for i in range(number_polys):
            if i % 6 == 0:
                f.write('          ')
            f.write(f'{(i + 1) * 4}')
            if i % 6 == 5:
                f.write('\n')
            else:
                f.write(' ')
        if number_polys % 6 != 0:
            f.write('\n')

        f.write('        </DataArray>\n')
        f.write('      </Polys>\n')

        # Footer
        f.write('    </Piece>\n')
        f.write('  </PolyData>\n')
        f.write('</VTKFile>\n')
