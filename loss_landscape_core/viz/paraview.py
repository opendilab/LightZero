"""
Export loss landscape to ParaView VTP format for high-quality 3D rendering.

VTP (VTK XML PolyData) format can be opened with ParaView for professional
visualization with lighting, shading, and high-resolution rendering.
"""

import math
import h5py
import numpy as np
from scipy import interpolate


def h5_to_vtp(surf_file, surf_name='train_loss', log=False, zmax=-1, interp=-1, output=None):
    """Convert HDF5 loss surface to VTP format for ParaView.

    Args:
        surf_file: Path to HDF5 surface file
        surf_name: Name of surface to export (default: 'train_loss')
        log: Apply logarithmic scale to z values
        zmax: Maximum z value (values above clipped to this)
        interp: Interpolate surface to higher resolution (-1: no interpolation)
        output: Output file path (default: auto-generated from surf_file)
    """
    print('-' * 60)
    print('Converting HDF5 to VTP format for ParaView')
    print('-' * 60)

    # Read HDF5 file
    f = h5py.File(surf_file, 'r')

    [xcoordinates, ycoordinates] = np.meshgrid(f['xcoordinates'][:], f['ycoordinates'][:])
    vals = f[surf_name][:]

    x_array = xcoordinates[:].ravel()
    y_array = ycoordinates[:].ravel()
    z_array = vals.ravel()

    # Interpolate to higher resolution if requested
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

    # Generate output filename
    if output is None:
        output = surf_file + f"_{surf_name}"
        if zmax > 0:
            output += f"_zmax={zmax}"
        if log:
            output += "_log"
        output += ".vtp"

    # Clip z values if requested
    if zmax > 0:
        z_array[z_array > zmax] = zmax

    # Apply log scale if requested
    if log:
        z_array = np.log(z_array + 0.1)

    print(f"Output file: {output}")

    # Prepare data
    number_points = len(z_array)
    matrix_size = int(math.sqrt(number_points))
    poly_size = matrix_size - 1
    number_polys = poly_size * poly_size

    print(f"Number of points: {number_points}")
    print(f"Matrix size: {matrix_size} x {matrix_size}")
    print(f"Number of polygons: {number_polys}")

    # Calculate statistics
    min_val = min(min(x_array), min(y_array), min(z_array))
    max_val = max(max(x_array), max(y_array), max(z_array))

    # Calculate averaged z values for cell data
    averaged_z_values = []
    for col in range(poly_size):
        stride = col * matrix_size
        for row in range(poly_size):
            idx = stride + row
            avg_z = (z_array[idx] + z_array[idx + 1] +
                     z_array[idx + matrix_size] + z_array[idx + matrix_size + 1]) / 4.0
            averaged_z_values.append(avg_z)

    # Write VTP file
    _write_vtp_file(output, x_array, y_array, z_array, averaged_z_values,
                    number_points, matrix_size, poly_size, number_polys,
                    min_val, max_val)

    f.close()
    print("Done!")


def _write_vtp_file(vtp_file, x_array, y_array, z_array, averaged_z_values,
                    number_points, matrix_size, poly_size, number_polys,
                    min_val, max_val):
    """Write VTP file in VTK XML PolyData format.

    Args:
        All parameters are internal to the conversion process
    """
    with open(vtp_file, 'w') as f:
        # Header
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n')
        f.write('  <PolyData>\n')
        f.write(f'    <Piece NumberOfPoints="{number_points}" NumberOfVerts="0" NumberOfLines="0" '
                f'NumberOfStrips="0" NumberOfPolys="{number_polys}">\n')

        # Point Data (z values)
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
