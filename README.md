# Danbooru Autotagger

A tag prediction system for anime-style images.

![image](https://user-images.githubusercontent.com/8430473/176574544-d8ebe9e0-fdf2-4090-8864-b856ce5e3ff9.png)

The current image uses
[animetimm/convnextv2_huge.dbv4-full](https://huggingface.co/animetimm/convnextv2_huge.dbv4-full),
a 12,476-tag ConvNeXt V2 model trained on Danbooru-style data. The container
ships a pinned FP16 OpenVINO conversion for Intel CPU and Xe GPU inference.

# Demo

Try it at https://autotagger.donmai.us.

Or go to https://danbooru.donmai.us/ai_tags to browse predicted tags on all posts on
Danbooru. Here are some examples of different tags:

* https://danbooru.donmai.us/ai_tags?search[tag_name]=comic&search[order]=score_desc
* https://danbooru.donmai.us/ai_tags?search[tag_name]=hatsune_miku&search[order]=score_desc
* https://danbooru.donmai.us/ai_tags?search[tag_name]=cat&search[order]=score_desc

# Quickstart

```
# Get tags for a single image
cat image.jpg | docker run --rm -i ghcr.io/danbooru/autotagger autotag -

# Run the web server. Open http://localhost:5000.
docker run --rm -p 5000:5000 ghcr.io/danbooru/autotagger

# Get tags from the web server.
curl http://localhost:5000/evaluate -X POST -F file=@hatsune_miku.jpg -F format=json
```

# Web

Start the app server:

```bash
# With Docker
docker run --rm -p 5000:5000 ghcr.io/danbooru/autotagger

# Without Docker
uv run gunicorn
```

Then open http://localhost:5000 to use the webapp. Here you can upload images and
view the list of predicted tags.

# API

Start the app server as above, then do:

```bash
curl http://localhost:5000/evaluate -X POST -F file=@hatsune_miku.jpg -F format=json
```

The output will look like this:

```json
[
  {
    "filename": "hatsune_miku.jpg",
    "tags": {
      "1girl": 0.9995526671409607,
      "hatsune_miku": 0.9995216131210327,
      "vocaloid": 0.9981155395507812,
      "solo": 0.9938727617263794,
      "thighhighs": 0.970325767993927,
      "long_hair": 0.9630335569381714,
      "twintails": 0.9352861046791077,
      "very_long_hair": 0.8532902002334595,
      "necktie": 0.8532789945602417,
      "aqua_hair": 0.8266996145248413,
      "detached_sleeves": 0.796751081943512,
      "skirt": 0.7879447340965271,
      "rating:s": 0.7843148112297058,
      "aqua_eyes": 0.6136178374290466,
      "zettai_ryouiki": 0.5611224174499512,
      "thigh_boots": 0.37453025579452515,
      "black_legwear": 0.37255123257637024,
      "full_body": 0.3261113464832306,
      "simple_background": 0.28789788484573364,
      "boots": 0.286143958568573,
      "headset": 0.27902844548225403,
      "white_background": 0.23441512882709503,
      "shirt": 0.21720334887504578,
      "looking_at_viewer": 0.2044636756181717,
      "pleated_skirt": 0.17705336213111877,
      "smile": 0.17575393617153168,
      "bare_shoulders": 0.17370294034481049,
      "headphones": 0.16347116231918335,
      "standing": 0.15511766076087952,
      "rating:g": 0.13711321353912354,
      "aqua_necktie": 0.11798079311847687,
      "black_skirt": 0.11197035759687424,
      "blush": 0.10813453793525696
    }
  }
]
```

# CLI

Generate tags for a single image:

```bash
# With Docker:
cat image.jpg | docker run --rm ghcr.io/danbooru/autotagger autotag -

# Without Docker:
./autotag image.jpg
```

Generate tags for multiple images:

```bash
# With Docker:
# `-v $PWD:/host` means mount the current directory as /host inside the Docker container.
docker run --rm -v $PWD:/host ghcr.io/danbooru/autotagger autotag /host/image1.jpg /host/image2.jpg

# Without Docker:
./autotag image1.jpg image2.jpg
```

Generate tags for all images inside the `images/` directory:

```bash
# With Docker:
# Change `images` to whatever your image directory is called.
docker run --rm -v $PWD/images:/images ghcr.io/danbooru/autotagger autotag /images

# Without Docker:
./autotag images/
```

Generate tags for all files inside a directory matching a pattern:

```bash
find images/ -name '*.jpg' | ./autotag -i -
```

Generate a list of tags in CSV format, suitable for importing into your own Danbooru instance:

```bash
./autotag -c -f -N images/ | gzip > tags.csv.gz
```

# Manual Installation

```bash
git clone https://github.com/danbooru/autotagger.git
cd autotagger
uv sync
uv run ./autotag test/hatsune_miku.jpg
```

The default local ONNX model is downloaded from Hugging Face on first use and
does not require a token. The original PyTorch repository is gated; the
container uses the public, pinned ONNX conversion linked below.

# Intel Xe GPU

OpenVINO uses automatic device selection by default. To require an Intel GPU
and fail instead of falling back to CPU:

```bash
docker run --rm \
  --device=/dev/dri \
  --group-add="$(stat -c '%g' /dev/dri/renderD* | head -n1)" \
  -e AUTOTAGGER_DEVICE=GPU \
  -p 5000:5000 \
  ghcr.io/danbooru/autotagger
```

In Kubernetes, expose the Intel GPU device to the pod (for example with the
Intel GPU DRA driver) and set `AUTOTAGGER_DEVICE=GPU`. Useful runtime settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_PATH` | bundled model | Hugging Face repo ID, model directory, or direct `.xml`/`.onnx` path |
| `MODEL_REVISION` | repository default | Optional Hugging Face revision for runtime downloads |
| `AUTOTAGGER_DEVICE` | `AUTO` | OpenVINO device such as `GPU`, `CPU`, or `AUTO` |
| `AUTOTAGGER_PERFORMANCE_HINT` | `LATENCY` | OpenVINO performance hint |
| `AUTOTAGGER_OPENVINO_CACHE_DIR` | `/tmp/autotagger-openvino-cache` | Compiled-model cache |

Use one Gunicorn worker unless the node has enough memory for multiple copies
of the model. Gunicorn threads share one model instance safely.

# Implementation

The build downloads the pinned
[`itterative/convnextv2_huge.dbv4-full-onnx`](https://huggingface.co/itterative/convnextv2_huge.dbv4-full-onnx)
conversion of the requested model and converts it to an OpenVINO IR with FP16
weights. This reduces model storage by roughly half and lets Intel Xe execute
the graph in FP16. The upstream tag vocabulary is checksum-verified during the
build.

Images are resized without distortion, centered on a white square, normalized
with ImageNet statistics, and passed to the model as RGB NCHW tensors. Model
logits are converted to probabilities with sigmoid. Rating labels retain the
service's existing `rating:g`, `rating:s`, `rating:q`, and `rating:e`
convention.

The application source is MIT licensed. The bundled model is GPL-3.0 licensed;
see the upstream model card for its terms.

# See also

* https://github.com/KichangKim/DeepDanbooru
* https://github.com/SmilingWolf/SW-CV-ModelZoo
* https://github.com/zyddnys/RegDeepDanbooru
* https://github.com/rezoo/illustration2vec
* https://www.gwern.net/Danbooru2021
* https://console.cloud.google.com/storage/browser/danbooru_public/data?project=danbooru1 (Danbooru data dumps)
