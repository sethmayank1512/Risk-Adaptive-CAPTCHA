## Quick start
```bash
pip install -r requirements.txt

# 1. get the real data
git clone https://github.com/chrisgdt/DELBOT-Mouse data/DELBOT-Mouse
cd data/DELBOT-Mouse/python && tar -xzf dataset.tar.gz && cd ../../..   
pip install huggingface_hub
hf download Capycap-AI/CaptchaSolve30k --repo-type dataset --local-dir data/CaptchaSolve30k --include "validation.jsonl"   

# 2. LOOK at the data (prints raw first lines + what the loaders detected)
python -m training.inspect_data --delbot data/DELBOT-Mouse/python/sessions.json \
                                --capycap data/CaptchaSolve30k/validation.jsonl

# 3. train + evaluate -> artifacts/ and reports/report.md (+ png plots)
python -m training.train --delbot data/DELBOT-Mouse/python/sessions.json \
                         --capycap data/CaptchaSolve30k/validation.jsonl

# 4. run the app, open http://127.0.0.1:5000
python app.py
```





## Layout
`app.py` · `risk/{features,model,engine,challenges}.py` · `static/collector.js` · `templates/index.html` ·
`training/{loaders,synth,train,inspect_data}.py` · `tools/{bots,summarize_collected}.py` · `tests/`


## Datasets (not included in this repository)
- DELBOT-Mouse: https://github.com/chrisgdt/DELBOT-Mouse (Apache 2.0)
- CaptchaSolve30k: https://huggingface.co/datasets/Capycap-AI/CaptchaSolve30k (Apache 2.0)

Download them into `data/` using the commands in the Quick start section above.
