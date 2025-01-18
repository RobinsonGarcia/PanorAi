### Homogeneous Coordinates and Gnomonic Projection

Homogeneous coordinates provide a powerful framework for representing geometric transformations, particularly when dealing with projections like the gnomonic projection. They allow for a unified treatment of translations, rotations, and perspective transformations, making them especially useful in deriving and interpreting projections. Here, the derivation of the gnomonic projection is revisited through the lens of homogeneous coordinates.

---

## 1. Homogeneous Representation of Points

### 1.1. Sphere Points

A point \(P\) on the sphere in Cartesian coordinates is represented in homogeneous coordinates as:
$$
P_h = (x, y, z, 1)^T,
$$
where the extra coordinate facilitates projective transformations.

In spherical coordinates, the point \(P\) is:
$$
P = (\cos\phi \cos\lambda, \cos\phi \sin\lambda, \sin\phi),
$$
where:
- $\phi$: Latitude of $P$,
- $\lambda$: Longitude of $P$.

The constraint for the sphere is:
$$
x^2 + y^2 + z^2 = R^2.
$$

### 1.2. Tangent Plane and Projection Center

- **Tangent Plane**: The tangent plane at the point \(P_0\) is defined by the normal vector \(\mathbf{n}\) and offset \(d\). Its equation in homogeneous coordinates is:
$$
\mathbf{n} \cdot P_h = d,
$$
where:
$$
\mathbf{n} = (\cos\phi_0 \cos\lambda_0, \cos\phi_0 \sin\lambda_0, \sin\phi_0),
$$
and:
$$
d = R.
$$

- **Projection Center**: The projection originates from the sphere’s center \((0, 0, 0)\), represented in homogeneous coordinates as:
$$
C_h = (0, 0, 0, 1)^T.
$$

---

## 2. Projection Line in Homogeneous Coordinates

The line passing through the sphere center and a point \(P\) on the sphere is parameterized in homogeneous form as:
$$
\mathbf{L}(t) = t \cdot P_h,
$$
where \(t\) is a scalar parameter.

---

## 3. Intersection with the Tangent Plane

The intersection of the line \(\mathbf{L}(t)\) with the tangent plane occurs when:
$$
\mathbf{n} \cdot \mathbf{L}(t) = d.
$$

Substituting \(\mathbf{L}(t) = t \cdot P_h\), we have:
$$
t \cdot (\mathbf{n} \cdot P_h) = d.
$$

Since \(\mathbf{n} \cdot P_h\) simplifies to:
$$
\mathbf{n} \cdot P_h = \cos\phi_0 \cos\phi \cos(\lambda - \lambda_0) + \sin\phi_0 \sin\phi,
$$
we solve for \(t\):
$$
t = \frac{d}{\cos\phi_0 \cos\phi \cos(\lambda - \lambda_0) + \sin\phi_0 \sin\phi}.
$$

Define:
$$
\cos c = \cos\phi_0 \cos\phi \cos(\lambda - \lambda_0) + \sin\phi_0 \sin\phi,
$$
where \(\cos c\) represents the cosine of the angular distance \(c\) between \(P_0\) and \(P\). Then:
$$
t = \frac{R}{\cos c}.
$$

---

## 4. Projected Coordinates on the Plane

Using \(t\), the projected point on the tangent plane in homogeneous coordinates is:
$$
P_{\text{proj}} = t \cdot P_h = \frac{R}{\cos c} (\cos\phi \cos\lambda, \cos\phi \sin\lambda, \sin\phi, 1)^T.
$$

Ignoring the \(z\)-coordinate, the 2D coordinates \((x, y)\) on the tangent plane are:
$$
x = R \frac{\cos\phi \sin(\lambda - \lambda_0)}{\cos c},
$$
$$
y = R \frac{\cos\phi_0 \sin\phi - \sin\phi_0 \cos\phi \cos(\lambda - \lambda_0)}{\cos c}.
$$

---

## 5. Homogeneous Form of Inverse Gnomonic Projection

The inverse projection recovers spherical coordinates \((\phi, \lambda)\) from plane coordinates \((x, y)\).

### 5.1. Radial Distance

The radial distance \(\rho\) is:
$$
\rho = \sqrt{x^2 + y^2}.
$$

### 5.2. Angular Distance

The angular distance \(c\) is:
$$
c = \arctan\left(\frac{\rho}{R}\right).
$$

### 5.3. Latitude \((\phi)\)

The latitude is derived as:
$$
\phi = \arcsin\left(\cos c \sin\phi_0 + \frac{y \sin c \cos\phi_0}{\rho}\right).
$$

### 5.4. Longitude \((\lambda)\)

The longitude is:
$$
\lambda = \lambda_0 + \arctan\left(\frac{x \sin c}{\rho \cos\phi_0 \cos c - y \sin\phi_0 \sin c}\right).
$$

---

## Summary of Homogeneous Gnomonic Projection Equations

### Forward Projection:
$$
x = R \frac{\cos\phi \sin(\lambda - \lambda_0)}{\cos c},
$$
$$
y = R \frac{\cos\phi_0 \sin\phi - \sin\phi_0 \cos\phi \cos(\lambda - \lambda_0)}{\cos c}.
$$

### Inverse Projection:
$$
\phi = \arcsin\left(\cos c \sin\phi_0 + \frac{y \sin c \cos\phi_0}{\rho}\right),
$$
$$
\lambda = \lambda_0 + \arctan\left(\frac{x \sin c}{\rho \cos\phi_0 \cos c - y \sin\phi_0 \sin c}\right).
$$


### Step 1: Projection Line Equation
The projection line is parameterized as:
$$
\mathbf{r}(t) = t \cdot P,
$$
where \(P = (\cos\phi \cos\lambda, \cos\phi \sin\lambda, \sin\phi)\) is a point on the sphere.

Substitute \(t = \frac{R}{\cos c}\) (from the previous derivation):
$$
\mathbf{r}(t) = \frac{R}{\cos c} (\cos\phi \cos\lambda, \cos\phi \sin\lambda, \sin\phi).
$$

The point \(\mathbf{r}(t)\) lies on the tangent plane at \(P_0\), and we now compute the projection in terms of the plane's coordinate system.

---

### Step 2: Tangent Plane Coordinate System
The tangent plane at \(P_0 = (\cos\phi_0 \cos\lambda_0, \cos\phi_0 \sin\lambda_0, \sin\phi_0)\) is defined with a local coordinate system:
- The \(x\)-axis is aligned with increasing longitude \(\lambda\),
- The \(y\)-axis is aligned with increasing latitude \(\phi\).

To express the projected coordinates \((x, y)\), we decompose \(\mathbf{r}(t)\) in terms of this local coordinate system.

---

### Step 3: Decomposing the Projection onto the Plane
1. **Relative longitude difference:**
   The longitude difference between \(P\) and \(P_0\) is:
   $$
   \lambda - \lambda_0.
   $$
   Using the spherical coordinate representation:
   - The term \(\sin(\lambda - \lambda_0)\) represents the horizontal component (relative to \(\lambda_0\)) of the projection.

2. **Horizontal (x-coordinate):**
   The \(x\)-coordinate in the plane is proportional to the horizontal displacement from \(P_0\). This comes from the spherical representation:
   $$
   x = R \frac{\cos\phi \sin(\lambda - \lambda_0)}{\cos c}.
   $$
   This follows from the fact that:
   - \(\cos\phi\) scales the longitude effect based on latitude,
   - \(\sin(\lambda - \lambda_0)\) isolates the relative displacement in longitude,
   - The division by \(\cos c\) adjusts for the distortion caused by the angular distance \(c\).

3. **Vertical (y-coordinate):**
   The \(y\)-coordinate in the plane accounts for the relative latitude difference. Starting from the spherical representation:
   $$
   y = R \frac{\cos\phi_0 \sin\phi - \sin\phi_0 \cos\phi \cos(\lambda - \lambda_0)}{\cos c}.
   $$
   Here:
   - \(\cos\phi_0 \sin\phi\): The contribution of the point's latitude relative to the tangent point,
   - \(-\sin\phi_0 \cos\phi \cos(\lambda - \lambda_0)\): The correction term accounting for the projection of the latitude onto the plane's \(y\)-axis,
   - \(\cos c\): Corrects for the distortion caused by the angular distance \(c\).

---

### Step 4: The Role of \(\sin(\lambda - \lambda_0)\)
The term \(\sin(\lambda - \lambda_0)\) appears in the \(x\)-coordinate because it directly reflects the relative displacement in longitude between the point being projected and the projection center \(P_0\). In spherical geometry:
- Longitude differences are not linear distances but are instead proportional to the sine of the angle between them, hence \(\sin(\lambda - \lambda_0)\).

In the tangent plane, the \(x\)-coordinate is effectively a scaled version of this longitude displacement, adjusted for the angular distance \(c\).

---

### Summary
- The \(x\)-coordinate (\(R \cos\phi \sin(\lambda - \lambda_0) / \cos c\)) arises from the horizontal displacement, which depends on the sine of the longitude difference.
- The \(y\)-coordinate combines vertical latitude displacement and corrections due to the longitude difference, projecting these components onto the tangent plane.

