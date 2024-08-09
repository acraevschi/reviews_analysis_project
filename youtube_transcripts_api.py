from youtube_transcript_api import YouTubeTranscriptApi
import regex as re
from bertopic import BERTopic  # reinitialize the conda environment

VIDEO_ID = "ZsBqnuRsmpI"

# def split_on_space_capital_letter(text):
#     pattern = r'\s(?=\p{Lu})'
#     return re.split(pattern, text)


transcript_lst = YouTubeTranscriptApi.list_transcripts(VIDEO_ID)

# need to automatize this part, namely, the determination of the language
for lang in transcript_lst:
    print(lang.language_code)

lang_code = lang.language_code
lang_name = lang.language.split()[0]

if lang_code != "en":
    pass  # TODO: call deepl API to translate the transcript

transcript = YouTubeTranscriptApi.get_transcript(VIDEO_ID, languages=[lang_code])

full_transcript = " ".join([item["text"] for item in transcript])
