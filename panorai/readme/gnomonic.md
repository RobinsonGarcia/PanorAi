# Gnomonic Projection: Step-by-Step Derivation

Gnomonic projection is a type of map projection where points on the surface of a sphere are projected from the center of the sphere onto a tangent plane. Below is the detailed mathematical derivation.

---

## 1. Define the Sphere and Coordinate System
- The sphere has radius \( R \) and is centered at the origin \( O \) in a Cartesian coordinate system \((x, y, z)\).
- The sphere is defined by:

$$
x^2 + y^2 + z^2 = R^2.
$$

- Points on the sphere are parameterized using latitude \( \phi \) (angle from the equator) and longitude \( \lambda \) (angle from the prime meridian):

$$
x = R \cos\phi \cos\lambda, \quad y = R \cos\phi \sin\lambda, \quad z = R \sin\phi.
$$

---

## 2. Select the Tangent Plane
- Assume the tangent plane touches the sphere at the North Pole (\( \phi = \frac{\pi}{2}, \lambda = 0 \)).
- The tangent plane is at:

$$
z = R.
$$

---

## 3. Projection Process
The projection maps points on the sphere \( P_h \) to points on the tangent plane \( P_{\text{proj}} \) by extending a line from the center of the sphere \( O \) through \( P_h \).

---

## 4. Find the Intersection with the Tangent Plane
- A line through the center \( O = (0, 0, 0) \) and a point \( P_h = (x, y, z) \) on the sphere is parameterized as:

$$
L(t) = t \cdot P_h = t \cdot (x, y, z).
$$

- The line intersects the tangent plane at \( z = R \). Solving for \( t \):

$$
t z = R \implies t = \frac{R}{z}.
$$

- Substituting \( t = \frac{R}{z} \) into \( L(t) \), the projected point \( P_{\text{proj}} \) on the tangent plane is:

$$
P_{\text{proj}} = t \cdot P_h = \frac{R}{z} (x, y, z).
$$

---

## 5. Simplify to Planar Coordinates
- The tangent plane is at \( z = R \), so we only consider the \( x \) and \( y \) coordinates of \( P_{\text{proj}} \):

$$
x' = \frac{R x}{z}, \quad y' = \frac{R y}{z}.
$$

- Substituting the spherical coordinates \( x = R \cos\phi \cos\lambda \), \( y = R \cos\phi \sin\lambda \), and \( z = R \sin\phi \):

$$
x' = \frac{R (R \cos\phi \cos\lambda)}{R \sin\phi} = \frac{R \cos\phi \cos\lambda}{\sin\phi},
$$

$$
y' = \frac{R (R \cos\phi \sin\lambda)}{R \sin\phi} = \frac{R \cos\phi \sin\lambda}{\sin\phi}.
$$

---

## 6. Final Formula for Gnomonic Projection
The planar coordinates \((x', y')\) in the gnomonic projection are:

$$
x' = R \cot\phi \cos\lambda,
$$

$$
y' = R \cot\phi \sin\lambda,
$$

where \( \cot\phi = \frac{\cos\phi}{\sin\phi} \).

---

## 7. Interpret the Projection
- **Straight lines** on the projection correspond to **great circles** on the sphere (the shortest path between two points).
- **Distortion** increases as you move away from the tangent point (e.g., the North Pole). Near the edges of the map, areas and angles are highly distorted.

---

# Generalized Gnomonic Projection for Any Tangent Plane

This is a continuation of the gnomonic projection derivation, expanded to work for any tangent plane defined by its latitude (\( \phi_0 \)) and longitude (\( \lambda_0 \)).

---

## 1. Define the Tangent Point and Orientation
- Let the tangent point on the sphere be defined by:
  - Latitude \( \phi_0 \) (northward positive).
  - Longitude \( \lambda_0 \) (eastward positive).

- The Cartesian coordinates of the tangent point \( P_0 \) are:
  $$
  P_0 = (x_0, y_0, z_0) = (R \cos\phi_0 \cos\lambda_0, R \cos\phi_0 \sin\lambda_0, R \sin\phi_0).
  $$

---

## 2. Coordinate Transformation
To generalize the projection, we transform the sphere so that the tangent point \( P_0 \) becomes the North Pole (\( \phi' = 90^\circ \)) in a new coordinate system. This involves two rotations:

1. **First rotation**: Rotate the sphere around the \( z \)-axis by \( -\lambda_0 \), so the tangent point lies on the \( xz \)-plane.
2. **Second rotation**: Rotate around the \( y \)-axis by \( -\phi_0 \), so the tangent point aligns with the new North Pole.

---

### Transformation Equations
Let the original spherical coordinates of a point \( P \) on the sphere be \( (\phi, \lambda) \). The Cartesian coordinates of \( P \) in the original system are:
$$
(x, y, z) = (R \cos\phi \cos\lambda, R \cos\phi \sin\lambda, R \sin\phi).
$$

#### 1. Rotation around the \( z \)-axis:
After rotating by \( -\lambda_0 \):
$$
x_1 = x \cos\lambda_0 + y \sin\lambda_0,
$$
$$
y_1 = -x \sin\lambda_0 + y \cos\lambda_0,
$$
$$
z_1 = z.
$$

#### 2. Rotation around the \( y \)-axis:
After rotating by \( -\phi_0 \):
$$
x' = x_1 \cos\phi_0 - z_1 \sin\phi_0,
$$
$$
y' = y_1,
$$
$$
z' = x_1 \sin\phi_0 + z_1 \cos\phi_0.
$$

---

## 3. Gnomonic Projection in the New System
The gnomonic projection maps the sphere onto a tangent plane at \( z' = R \) in the new coordinate system. The planar coordinates are:
$$
x_{\text{proj}} = \frac{R x'}{z'},
$$
$$
y_{\text{proj}} = \frac{R y'}{z'}.
$$

---

## 4. Substitute Back to Original Coordinates
Using the transformed coordinates \( x', y', z' \), substitute the original spherical coordinates \( (\phi, \lambda) \). After simplification, the planar coordinates in terms of the original spherical coordinates and the tangent point \( (\phi_0, \lambda_0) \) are:

### Generalized Gnomonic Projection Equations:
$$
x_{\text{proj}} = \frac{R \cos\phi \sin(\lambda - \lambda_0)}{\sin\phi \sin\phi_0 + \cos\phi \cos\phi_0 \cos(\lambda - \lambda_0)},
$$
$$
y_{\text{proj}} = \frac{R (\cos\phi_0 \sin\phi - \sin\phi_0 \cos\phi \cos(\lambda - \lambda_0))}{\sin\phi \sin\phi_0 + \cos\phi \cos\phi_0 \cos(\lambda - \lambda_0)}.
$$

---

## 5. Interpretation of the Generalized Projection
1. **Tangent Point**: The tangent point \( (\phi_0, \lambda_0) \) maps to the origin of the projection plane \( (0, 0) \).
2. **Straight Lines**: Great circles passing through the tangent point are straight lines on the projection.
3. **Distortion**: Distortion increases as the distance from the tangent point increases. At large distances, the denominator \( \sin\phi \sin\phi_0 + \cos\phi \cos\phi_0 \cos(\lambda - \lambda_0) \) approaches zero, leading to infinite distortion.

---

## 6. Special Cases
1. **North Pole as Tangent Point**: If \( \phi_0 = 90^\circ \), the equations reduce to the standard gnomonic projection:
   $$
   x_{\text{proj}} = R \cot\phi \cos\lambda,
   $$
   $$
   y_{\text{proj}} = R \cot\phi \sin\lambda.
   $$
2. **Equatorial Tangent Plane**: If \( \phi_0 = 0^\circ \), the equations simplify for tangent planes centered on the equator.

This generalization allows the gnomonic projection to be applied to any tangent plane defined by arbitrary latitude and longitude.

---
# Inverse Mapping for Generalized Gnomonic Projection

To recover the spherical coordinates \( (\phi, \lambda) \) from planar coordinates \( (x_{\text{proj}}, y_{\text{proj}}) \), follow the steps below.

---

## 1. Forward Mapping Recap

### Generalized Gnomonic Projection Equations:
$$
x_{\text{proj}} = \frac{R \cos\phi \sin(\lambda - \lambda_0)}{\sin\phi \sin\phi_0 + \cos\phi \cos\phi_0 \cos(\lambda - \lambda_0)},
$$
$$
y_{\text{proj}} = \frac{R (\cos\phi_0 \sin\phi - \sin\phi_0 \cos\phi \cos(\lambda - \lambda_0))}{\sin\phi \sin\phi_0 + \cos\phi \cos\phi_0 \cos(\lambda - \lambda_0)}.
$$

Here:
- \( \phi_0 \) and \( \lambda_0 \) are the latitude and longitude of the tangent point.
- \( R \) is the sphere's radius.

---

## 2. Normalize the Planar Coordinates
Normalize the planar coordinates \( (x_{\text{proj}}, y_{\text{proj}}) \) to remove the dependence on \( R \):
$$
X = \frac{x_{\text{proj}}}{R}, \quad Y = \frac{y_{\text{proj}}}{R}.
$$

---

## 3. Compute the Auxiliary Angle \( c \)
The auxiliary angle \( c \), which represents the angular distance between the tangent point \( (\phi_0, \lambda_0) \) and the projected point, is calculated as:
$$
\tan c = \sqrt{X^2 + Y^2}.
$$
Thus:
$$
c = \arctan(\sqrt{X^2 + Y^2}).
$$

---

## 4. Latitude (\( \phi \))
The latitude \( \phi \) is derived from the following relationship:
$$
\sin\phi = \sin\phi_0 \cos c + Y \sin c \cos\phi_0,
$$
$$
\cos\phi = \sqrt{1 - \sin^2\phi}.
$$

---

## 5. Longitude (\( \lambda \))
The longitude \( \lambda \) is computed using:
$$
\sin(\lambda - \lambda_0) = \frac{X \sin c}{\sin\phi},
$$
$$
\cos(\lambda - \lambda_0) = \frac{\cos c - \sin\phi \sin\phi_0}{\cos\phi \cos\phi_0}.
$$

Finally:
$$
\lambda = \lambda_0 + \arctan2\left(\sin(\lambda - \lambda_0), \cos(\lambda - \lambda_0)\right).
$$

---

## 6. Summary of the Process

1. Normalize the planar coordinates:
   $$
   X = \frac{x_{\text{proj}}}{R}, \quad Y = \frac{y_{\text{proj}}}{R}.
   $$

2. Compute the auxiliary angle \( c \):
   $$
   c = \arctan(\sqrt{X^2 + Y^2}).
   $$

3. Compute the latitude \( \phi \):
   $$
   \sin\phi = \sin\phi_0 \cos c + Y \sin c \cos\phi_0,
   $$
   $$
   \cos\phi = \sqrt{1 - \sin^2\phi}.
   $$

4. Compute the longitude \( \lambda \):
   $$
   \sin(\lambda - \lambda_0) = \frac{X \sin c}{\sin\phi},
   $$
   $$
   \cos(\lambda - \lambda_0) = \frac{\cos c - \sin\phi \sin\phi_0}{\cos\phi \cos\phi_0}.
   $$
   $$
   \lambda = \lambda_0 + \arctan2\left(\sin(\lambda - \lambda_0), \cos(\lambda - \lambda_0)\right).
   $$

This process recovers the spherical coordinates \( (\phi, \lambda) \) from the planar coordinates \( (x_{\text{proj}}, y_{\text{proj}}) \).