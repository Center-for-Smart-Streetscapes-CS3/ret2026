# CS3 AI Teaching Kit: Ubuntu installation

The offline CS3 AI Teaching Kit is published as a Hugging Face dataset:

<https://huggingface.co/datasets/mehmetkeremturkcan/cs3-ai-teaching-kit>

## Install or update the kit

Open a terminal on the Jetson or another Ubuntu computer and run:

```bash
sudo apt update
sudo apt install -y python3-venv

python3 -m venv "$HOME/.venvs/huggingface"
"$HOME/.venvs/huggingface/bin/pip" install --upgrade huggingface_hub

"$HOME/.venvs/huggingface/bin/hf" download \
  mehmetkeremturkcan/cs3-ai-teaching-kit \
  --repo-type dataset \
  --local-dir "$HOME/Desktop/cs3-teaching-kit" \
  --force-download
```

The same command handles the first installation and later updates. `--force-download` replaces packaged files with the newest published versions. Files on the Desktop that are not part of the dataset are left in place.

Close any lesson programs before updating. When the download finishes, open:

```text
~/Desktop/cs3-teaching-kit/index.html
```

The lesson index works offline in a browser. Enter a lesson folder before running its Python program because models and media are loaded with relative paths.

## Online component

The companion browser application contains 17 interactive lessons:

<https://center-for-smart-streetscapes-cs3.github.io/ret2026/toolkit/>

Learn more about both components:

<https://center-for-smart-streetscapes-cs3.github.io/ret2026/cs3-teaching-kit/>
