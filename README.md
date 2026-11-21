# Seerah Companion

A phone-friendly study page for the Seerah lecture series: the YouTube video plays inside the page,
with an accurate transcript that follows along, chapter summaries, lessons, and a glossary of
Arabic names and terms (transliteration + Arabic script).

- `index.html` – the whole website (hosted on GitHub Pages).
- `data/` – one JSON file per episode (`episodes/001.json` …), `index.json`, and the series `glossary.json`.
- `pipeline/process.py` – generates the data with the Gemini API, which watches the public YouTube
  video directly. Usage: `python3 pipeline/process.py data 3:VIDEO_ID 4:VIDEO_ID`
  (needs a Gemini API key sent as the `x-goog-api-key` header).

Qur'an text, transliteration and English translation in `pipeline/quran_data.json` come from the
[quran-json](https://github.com/risan/quran-json) project (Uthmani text from QuranEnc, transliteration from
Tanzil, Sahih International translation), licensed CC BY-SA 4.0.
