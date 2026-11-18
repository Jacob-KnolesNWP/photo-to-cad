# TrueShape

Phone photos of real parts, turned into CAD geometry at true scale, for the floor at Northland Wood Products. The person using it has SOLIDWORKS. The phone is the measuring instrument. A printed marker sheet is the ruler.

The home screen is the everyday path: download a marker sheet, drop in one or many photos (or a zip, including iPhone HEIC), enter thickness and an optional caliper reading, and get a DXF per photo plus a zip of all of them.

Two modes:

* **Flat part.** A plate, bracket, gasket, or template. The app straightens the photo, finds the outline, fits lines, arcs, and holes, and writes a 1:1 DXF in millimetres or inches.
* **3D part.** A walk around the object. COLMAP solves the cameras on the server. Pairwise stereo builds a mesh. Download STL or OBJ.

Flat parts that fit on one sheet are aimed at **under a millimetre**. 3D from a phone is about **±1.5 mm on a small object**, and the reported band grows with size. Critical dimensions still get checked with calipers. The app says so on the screen.

## Run it

Python 3.12, then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # only if you will run tests
uvicorn partcapture.main:app --host 0.0.0.0 --port 8000
```

Open `http://<this-computer>:8000` on the shop phone. Same Wi-Fi is enough. The phone camera is used directly (Chrome and Safari). You can also drop several photos at once, or a `.zip` of them. iPhone HEIC is accepted. Non-images inside a zip are ignored.

Jobs and downloads are stored in `./data` (override with `PARTCAPTURE_DATA`).

### Docker

COLMAP’s dense patch-match stereo needs an NVIDIA GPU. This image does not. Poses come from COLMAP on the CPU. The surface comes from OpenCV stereo. That is the honest 1–2 mm path, not a lab scanner.

```bash
docker compose up --build
```

The site is on port 8000. Job files persist in the `trueshape-data` volume.

## Print the marker sheets

The home screen has big download buttons: letter sheets 1–4 together, each letter sheet on its own, tabloid, and the 24×36 in board. The reminder on that screen is the whole rule: print at 100%, then check the black bar with a ruler.

| Sheet | Page | Markers | Scale bar | Use |
| --- | --- | --- | --- | --- |
| Letter | 8.5 × 11 in | 18 mm frame | 100 mm | Everyday parts, and a numbered set for large flat parts |
| Tabloid | 11 × 17 in | 22 mm frame | 200 mm | Bigger shop printer |
| Print-shop board | 24 × 36 in | 36 mm, frame plus a sparse interior grid | 400 mm | Send the PDF to a print shop |

Print settings that matter:

* **100% scale.** Turn off Fit to Page, Shrink to fit, and poster tiling.
* Check the black bar with a ruler. If a 100 mm bar does not measure 100 mm, reprint. Do not measure a part on a sheet that failed this check.
* Letter sheets are numbered. One download of several sheets gives each page its own marker IDs (up to 12 letter sheets, 5 tabloid, 4 large boards). Printing the same PDF twice duplicates IDs and the app cannot tell the copies apart.

## Thickness, backing paper, and a caliper

A part that stands up off the sheet looks bigger when the phone is close. An 86 × 40 mm remote, about 14 mm thick, photographed from about 200 mm, measured 88.2 × 41.2 mm with no thickness correction. With thickness 14 mm and a square (prism) edge it came back to 85.83 × 40.10 mm.

Enter the thickness on the home screen. **Square / prism** shrinks the silhouette by (camera height − thickness) / height, which is the footprint of a vertical wall. **Rounded** assumes the widest section is halfway up and uses half the thickness. The camera height comes from the markers (`solvePnP`) and the phone’s EXIF 35 mm-equivalent focal length when the photo still has it.

One caliper reading of the longest side then locks the scale. If that changes the size by more than 3%, the result warns you to recheck the print and the thickness.

A light part on coloured or dark paper is found automatically when that paper is a large rectangle inside the markers: the sheet colour is fit first, then the backing colour, then the part. A dark part on the white sheet needs no backing.

Warnings you should read before trusting a number: camera closer than 300 mm, tilt over 12°, a blurry photo, and the part within 2 mm of a marker.

A collapsible checklist on the home screen has the full shooting list (16–20 in, 2x lens, no flash, tape the sheet flat).

## Photograph a flat part

1. Sheet flat on the bench. Part flat on the sheet. Markers visible around the part, not covered.
2. Hold the phone roughly above the part. It does not have to be perfectly square — the markers straighten the photo.
3. Tap the sheet to focus. If the app says the photo is blurry, it is, and the outline would be wrong.
4. Compare the on-screen length, width, and hole diameters with calipers.
5. If the part is white or very light, auto-outline will miss it. Use **Touch up**.
6. Download the DXF. Units are real millimetres or inches (`$INSUNITS` is set). SOLIDWORKS imports the `OUTLINE` and `HOLES` layers. A note on the `NOTES` layer records the estimated accuracy.

### Large flat parts

Use several letter sheets, or one print-shop board, or both.

* **One large board, several photos.** Every marker on that board already has a known position. Overlap the photos. Each one needs at least four readable markers.
* **Several letter sheets around a part that does not fit on one page.** Lay them flat, not on top of each other. Take overlapping photos, and make sure at least one photo shows markers from two neighboring sheets at once (two markers on one of those sheets). That is how the sheets get tied into one coordinate system. If the groups never meet in a photo, the app stops and says so.
* Cover the bench between sheets with white paper, or plan to touch up. The auto-outline treats “not white paper” as the part, so a wooden bench in the gap becomes part of the outline.
* Do not cover the scale bar if you can help it. If the part does cover it, the app leaves that area alone instead of cutting a notch, as long as it looks like wood rather than black ink on white paper.

After the outline looks right, type one or two caliper readings (overall length, overall width, or both):

* One number scales the whole outline evenly. This is the right fix for a sheet that was printed a percent off, or a small residual scale error.
* Two numbers that agree within 0.8% still scale evenly, using the average.
* Two numbers that disagree by more than that stretch the outline to both readings. Holes can stop being perfect circles. The screen warns you.

The estimated accuracy has two parts. Capture quality is how well the markers agreed. The millimetre band also grows with the part:

```
sigma = sqrt(reprojection² + (0.12% of the diagonal)² + (0.20 mm × extra sheets)² + small photo and pixel terms)
```

A part on one letter sheet stays under 1 mm in this estimate. A 2 m diagonal is a few millimetres even with a clean capture. That growth is shown on the result screen, not hidden.

## Photograph a 3D part

1. Markers around the object, standing or lying so they show up in several photos. On a large piece, spread them out. Unique sheets (the numbered set) can sit on different sides.
2. At least 10 photos, 20 or more is safer, more still for anything over about a metre. Overlap each shot about two thirds with the previous one. Walk all the way around, then do a higher loop.
3. The object must not move. Smooth paint and shiny metal usually fail; wood grain is fine. Pencil marks or masking tape give the matcher something to see.
4. The model opens in the browser. Drag to turn it, pinch to zoom. Download STL or OBJ in millimetres.
5. Scale comes from the marker squares when the same marker is seen from two angles. If that fails, the model is **not** in real units until you set a scale:
   * Tap two points and type the real distance (a scale bar, or a span you taped).
   * Or type an overall width, depth, or height from calipers. One or two readings apply one scale factor. If they disagree, the median is used and the screen says so.

3D error estimate:

```
sigma = sqrt((1.5 mm)² + (0.30% of the diagonal)² + marker-disagreement term)
```

About ±1.5 mm on a small object. A 1 m diagonal is several millimetres. The result screen states this. Do not cut a critical dimension from the mesh alone.

## What the tests measured

Synthetic photos with known geometry, OpenCV ArUco, no printer and no bench curl:

| Case | Error |
| --- | --- |
| 100 × 60 mm plate, 16 mm hole, tilted pinhole photo | length 0.25 mm, width 0.10 mm, hole 0.08 mm |
| 86 × 40 mm prism, silhouette magnified as if shot from 200 mm with 14 mm thickness | corrected length and width within 1.5 mm |
| Light part on a coloured backing sheet | the part is measured, not the backing |
| 80 × 50 mm plate with mild lens distortion (k1 = −0.18) | 0.20 mm and 0.15 mm |
| 345 × 100 mm plate across two letter sheets and two photos | about 0.75 mm on length, 0.70 mm on width |
| 12-photo walk-around of a 100 × 70 × 50 mm block with upright markers | all photos aligned, scale from the markers, mesh in millimetres (scene bbox on the order of the block plus the markers, not an arbitrary SfM unit) |

Run them with:

```bash
pytest
```

The 3D test is the slow one. It is still only a few seconds on a CPU for the 640×480 sample.

## How the measurement works

**Flat.** ArUco markers from `DICT_5X5_1000`, each with a known position on its sheet. A homography maps the sheet plane to the photo. A one-parameter radial term is kept only when it clearly beats the plain homography. Several sheets are linked by photos that see more than one sheet, then solved as a 2D pose graph. Photos are resampled onto one metric mosaic (y up, millimetres). The part is whatever is not white paper, not a detected marker, and not a printed scale bar. The outline is split into lines and arcs; round holes become circles. DXF is written with ezdxf at 1:1.

**3D.** COLMAP SIFT, matching, and incremental mapping (CPU). Marker edges that show up in two registered photos set the scale. Upright markers also hint which way is up. Depth comes from stereo between neighboring cameras, fused into a voxel mesh (`marching_cubes`). COLMAP’s own dense stereo (`patch_match_stereo`) is CUDA-only and is not used.

## Limits

* Sub-millimetre flat results assume a sheet printed at 100%, lying flat, and a part that is not the same color as the paper. Touch-up is there for the cases that fail.
* The size term in the accuracy estimate is a shop allowance (residual lens error, paper not perfectly flat, print error). It is not a ISO certificate.
* Very large flat parts are limited by how well the sheets are tied together and by pixel size. The mosaic backs off from 10 px/mm so a big part still fits in memory, and the reported band grows.
* 3D on bare, shiny, or one-color surfaces often will not align. The app says to retake with more overlap and more visible grain.
* The 3D mesh is coarser than a metrology scanner. Expect to sand the numbers with calipers. Thin features and sharp inside corners get rounded by the voxels.
* Marker IDs are unique only inside one downloaded set. Reprinting an old PDF reuses IDs.
* One job runs at a time. A 3D job of 30 phone photos can take several minutes on a CPU.
