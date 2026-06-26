# Infeasibility-Algos
<i>generate_pointcloud, view_clusters :</i> generates pointcloud from obstacles, uses alpha reconstruction and saves the meshes as obj files <br><br>
<i>cspace_scattering:</i> Rays scatter from start/goal only, zero sdf points are found on rays using regula falsi. midpoints are calculated<br><br>
<i>cspace_3d_frontier:</i> rays are scattered from points inside -ve sdf region to create new points. they are joined to form graph structure. closedness cannot be interpreted directly<br><br>
<i>cspace_3d_hexagon:</i> uses rays from start/goal to create Hexagons. WORKS!<br><br>
<i>cspace_qhull:</i> uses cspace_3d_frontier but then applies QHULL over it. traces all tets on the surface of QHULL to get +ve and -ve sdf regions. WORKS!!<br><br>
<i>cspace_3d_boundary:</i> Original algorithm, boudnary tracing. can be used as ground truth<br><br>
<i>ground_truth_3d:</i> absolute ground truth.<br><br>
