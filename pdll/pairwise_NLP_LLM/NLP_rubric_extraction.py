import logging
from pathlib import Path

logger = logging.getLogger("result")

files_need_formatting = False

formatting_keywords = [
    "Prompt",
    "Rubric Guidelines",
    "Score",
    "Adjudication Rules",
    "Source Essay",
    "Conventions",
    "Sentence Fluency",
    "Word Choice",
    "Voice",
    "Organization",
    "Ideas",
    "Style",
    "Total Composite Score",
]


def get_rubric_texts_from_files() -> dict[int, str]:

    target_dir = Path("data-set/scoring_rubrics")

    file_dict = {}
    for file in target_dir.iterdir():
        if file.is_file() and file.suffix == ".txt":
            essay_set_ID = int(file.stem.split("_")[2])

            global files_need_formatting
            if files_need_formatting:
                format_rubric_text_file(file, formatting_keywords, "Essay Set")
                files_need_formatting = False

            with open(file, "r", encoding="utf-8") as rf:
                rubric_text = rf.read()

            file_dict[essay_set_ID] = rubric_text
    logger.debug("extracted rubric texts from corresponding files")
    return file_dict


def format_rubric_text_file(file: Path, keyword_list: list[str], filter: str):
    with open(file, "r", encoding="utf-8") as reader, open(
        file, "r+", encoding="utf-8"
    ) as writer:
        for line in reader:
            if line.rstrip() and not line.startswith(filter):
                for keyword in keyword_list:
                    if line.startswith(keyword):
                        writer.write("\n")
                writer.write(line.strip())
    logger.debug("formatted rubric text file")
