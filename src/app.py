import json
import os
import pathlib
import re
import sys
import traceback
from decimal import Decimal
from glob import glob
from typing import List, Tuple, Dict
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, \
    QPushButton, QTextEdit, QFileDialog, QProgressBar, QMessageBox, QComboBox
from PyQt5.QtCore import QThread, pyqtSignal, QSettings

import click
import openai
import tiktoken
from openai.error import RateLimitError, APIError
import backoff

_BATCH_SIZE = 10

_VERBOSITY = 1

_TRANSLATION_CACHE = {}
_TRANSLATION_CACHE_FILE = 'translation_cache.json'
_TEMPERATURE = 0.7

# Link https://beta.openai.com/docs/models/gpt-3
_DEFAULT_ENGINE = 'gpt-4o-mini'
_AVAILABLE_ENGINES = [
    'gpt-4o-mini',
    'gpt-4o'
]

# Link https://openai.com/pricing#language-models
_PRICING_PER_1K_TOKENS = {
    'gpt-4o-mini': 0.0001125,
    'gpt-4o': 0.01
}

# Link https://platform.openai.com/docs/models/overview
_TOKEN_LIMITS_PER_REQUEST = {
    'gpt-4o-mini': 128000,
    'gpt-4o': 128000
}


def set_verbosity(verbosity: int):
    if 0 < verbosity <= 3:
        globals()['_VERBOSITY'] = verbosity


def estimate_tokens(engine: str, text: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model(engine)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def estimate_cost(engine: str, tokens: int) -> Decimal:
    return Decimal(tokens / 1000 * _PRICING_PER_1K_TOKENS[engine]).quantize(Decimal('0.001'))


def contains_letters(text: str) -> bool:
    return bool(re.search(r'[a-zA-Z]', text))


def render_file(filepath: str) -> str:
    with open(str(pathlib.Path(filepath)), 'r', encoding='utf-8') as f:
        return f.read()


def calculate_tokens(engine: str, text: str) -> int:
    encoding = tiktoken.encoding_for_model(engine)
    num_tokens = len(encoding.encode(text))
    return num_tokens


def calculate_pricing(engine: str, tokens: int) -> int:
    pricing = Decimal((tokens / 1000) * _PRICING_PER_1K_TOKENS[engine])
    return round(pricing, 4)


def persist_cache():
    with open(_TRANSLATION_CACHE_FILE, 'w', encoding='utf-8') as f:
        print("Persisting translation cache...")
        json.dump(_TRANSLATION_CACHE, f, indent=4)


def verbose(level: int):
    import functools

    def actual_decorator(func):
        def neutered(*args, **kw):
            return

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return (
                func(*args, **kwargs)
                if level <= _VERBOSITY
                else neutered
            )

        return wrapper

    return actual_decorator


@verbose(level=3)
def debug(text, *args, **kwargs):
    click.echo(text, *args, **kwargs)


@verbose(level=2)
def info(text, *args, **kwargs):
    click.echo(text, *args, **kwargs)


@verbose(level=1)
def echo(text, *args, **kwargs):
    click.echo(text, *args, **kwargs)


class TranslationItem:
    def __init__(self, source_line=0, target_line=0, original_content="", suffix=""):
        self.source_line = source_line
        self.target_line = target_line
        self.translation_string = None
        self.original_content = original_content.replace(r'\"', '"')
        if original_content:
            self.translation_string = TranslationString(self.original_content)
        self.suffix = suffix

    def translate(self, engine, to_language):
        if self.translation_string:
            self.translation_string.translate(engine, to_language)

    def get_translated_content(self):
        if self.translation_string:
            return self.translation_string.translation.replace("\n", "\\n")
        return self.original_content

    def __repr__(self):
        return f"TranslationItem({self.source_line + 1}, {self.target_line + 1})"


class TranslationBlock:

    def __init__(self, source_file=None, block_line=0):
        self.source_file = source_file
        self.block_line = block_line
        self.translation_items = []

    def add_translation_item(self, translation_item):
        self.translation_items.append(translation_item)

    def translate(self, engine, to_language):
        for item in self.translation_items:
            item.translate(engine, to_language)

    def __iter__(self):
        return iter(self.translation_items)

    def __repr__(self):
        return "TranslationBlock({}, {})".format(self.source_file, self.block_line + 1)


class TranslationFile():

    def __init__(self, filename):
        self.filename = filename
        self.translation_blocks = []

    def add_translation_block(self, translation_block):
        self.translation_blocks.append(translation_block)

    def translate(self, engine, to_language):
        for block in self.translation_blocks:
            block.translate(engine, to_language)

    def __iter__(self):
        return iter(self.translation_blocks)

    def __repr__(self):
        return "TranslationFile({}, {} blocks)".format(self.filename, len(self.translation_blocks))


class TranslationString:
    def __init__(self, content):
        self.content = content
        self.engine = None
        self.to_language = None
        self.translation = None
        self.is_cached = False
        self.needs_translation = contains_letters(content)

    def translate(self, engine, to_language):
        print(f"Translating string: {self.content[:50]}...")
        self.engine = engine
        self.to_language = to_language

        if self.translation:
            return

        if not self.content.strip():
            self.translation = self.content
            self.is_cached = True
            return

        if not self.needs_translation:
            self.translation = self.content
            self.is_cached = True
            return

        cached_translation = self.pull_from_cache(to_language)
        if cached_translation:
            self.translation = cached_translation
            self.is_cached = True
            return

    def pull_from_cache(self, to_language):
        available_translations = _TRANSLATION_CACHE.get(self.content, None)
        if available_translations:
            cached_translation = available_translations.get(to_language, None)
            if cached_translation:
                return cached_translation

    def __repr__(self):
        return f'TranslationString(content="{self.content}", translation="{self.translation}", needs_translation={self.needs_translation})'


def batch_translate(strings: List[TranslationString], engine: str, to_language: str, api_base: str = None,
                    progress_callback=None):
    for i in range(0, len(strings), _BATCH_SIZE):
        batch = strings[i:i + _BATCH_SIZE]
        translate_batch(batch, engine, to_language, api_base)
        if progress_callback:
            progress_callback(min(i + _BATCH_SIZE, len(strings)))


def parse_translations(translated_text: str) -> List[str]:
    """
    Parse the translated text into individual translations.

    Args:
    translated_text (str): The full translated text returned by the API.

    Returns:
    List[str]: A list of individual translations.
    """
    # Split the translated text into individual blocks
    blocks = re.split(r'\n*---\n*', translated_text)

    # Remove any numbering or extra whitespace from each block
    cleaned_blocks = [re.sub(r'^\s*\[\d+\]\s*', '', block.strip()) for block in blocks]

    return cleaned_blocks


def parse_file(file: str) -> TranslationFile:
    """
    Parse a single file and extract translation blocks and items.

    Args:
    file (str): The path to the file to be parsed.

    Returns:
    TranslationFile: An object containing all translation blocks and items from the file.
    """
    translation_file = TranslationFile(file)
    translation_block = None
    translation_item = None

    with open(file, 'r', encoding='utf-8') as f:
        text_lines = f.readlines()
        i = 0
        while i < len(text_lines):
            line = text_lines[i].strip()

            # Check for the start of a new translation block
            if line.startswith("translate"):
                if translation_block:
                    translation_file.add_translation_block(translation_block)
                translation_block = TranslationBlock(source_file=file, block_line=i)
                i += 1
                continue

            # Check for voice line followed by dialogue
            if line.startswith('# voice'):
                i += 1
                if i < len(text_lines):
                    dialogue_line = text_lines[i].strip()
                    if dialogue_line.startswith('# '):
                        # Extract the dialogue content
                        dialogue_match = re.match(r'# (\w+) "(.*)"', dialogue_line)
                        if dialogue_match:
                            speaker, dialogue = dialogue_match.groups()
                            translation_item = TranslationItem(
                                source_line=i,
                                target_line=i + 2,
                                original_content=dialogue
                            )
                            translation_block.add_translation_item(translation_item)
                i += 1
                continue

            # Check for the start of a new translation block
            if re.match(r"translate ([A-z0-9_]+) ([A-z0-9_]+)", line):
                if translation_block:
                    translation_file.add_translation_block(translation_block)
                translation_block = TranslationBlock(source_file=file, block_line=i)
                i += 1
                continue

            # Check for old/new translation pairs
            if text_lines[i - 1].strip().startswith("#") and line.strip().startswith("old"):
                m = re.match(r'(\s*)(\w+)?\s*"(.*)"', line.strip())
                if m:
                    translation_item = TranslationItem(source_line=i, original_content=m.group(3))
                i += 1
                continue

            if text_lines[i - 1].strip().startswith("old") and line.strip().startswith("new"):
                m = re.match(r'(\s*)(\w+)?\s*"(.*)"', line.strip())
                if m:
                    translation_item.translation_string.translation = m.group(3)
                translation_item.target_line = i
                translation_block.add_translation_item(translation_item)
                i += 1
                continue

            # Check for inline translations
            if text_lines[i - 1].strip().startswith("#") \
                    and not text_lines[i - 1].strip().startswith("# nvl clear") \
                    and not text_lines[i - 1].strip().startswith("# voice") \
                    and not line.strip().startswith("voice ") \
                    and line.strip():
                text_match = re.match(r'# ?(\w+)? "(.*)"(.*)?', text_lines[i - 1].strip())
                if text_match:
                    speaker, text, suffix = text_match.groups()
                else:
                    i += 1
                    continue

                if line.strip().startswith("nvl clear"):
                    translation_item = TranslationItem(source_line=i,
                                                       original_content=text,
                                                       suffix=suffix,
                                                       target_line=i + 1)
                else:
                    translation_item = TranslationItem(source_line=i,
                                                       original_content=text,
                                                       suffix=suffix,
                                                       target_line=i)

                translation = re.match(r'.*"(.*)"', line.strip())
                if translation:
                    translation_item.translation_string.translation = translation.group(1)

                translation_block.add_translation_item(translation_item)
                i += 1
                continue
            i += 1

    # Add the last translation block if it exists
    if translation_block:
        translation_file.add_translation_block(translation_block)

    return translation_file


@backoff.on_exception(backoff.expo, (RateLimitError, APIError), max_tries=5)
def translate_batch(batch: List[TranslationString], engine: str, to_language: str, api_base: str = None):
    prefix = f'Translate the following text blocks to {to_language}. Keep all markers (especially square and curly brackets) and do not add new comments. Separate each translated block with "---". Here are the text blocks:'
    content = "\n".join([f"[{i + 1}] {item.content}" for i, item in enumerate(batch) if item.needs_translation])

    full_prompt = f"{prefix}\n\n{content}"

    try:
        response = openai.ChatCompletion.create(
            request_timeout=30,
            api_base=api_base,
            model=engine,
            messages=[
                {"role": "system", "content": "You are a translator."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=_TEMPERATURE,
            frequency_penalty=0,
            presence_penalty=0
        )
        translated_text = response.choices[0]['message']['content']
        translations = parse_translations(translated_text)

        translation_index = 0
        for item in batch:
            if item.needs_translation:
                item.translation = translations[translation_index].strip()
                _TRANSLATION_CACHE[item.content] = {to_language: item.translation}
                translation_index += 1
            else:
                item.translation = item.content

        persist_cache()
    except Exception as e:
        echo(f'Error in batch translation: {str(e)}')
        raise


class TranslationWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, api_key, translate_to, in_path, out_path, engine):
        super().__init__()
        self.api_key = api_key
        self.translate_to = translate_to
        self.in_path = in_path
        self.out_path = out_path
        self.engine = engine
        self.stop_requested = False

    def run(self):
        try:
            openai.api_key = self.api_key

            self.log.emit("Searching for files...")
            files = glob(os.path.join(self.in_path, "**", "*.rpy"), recursive=True)
            self.log.emit(f"Found {len(files)} files to process.")

            estimation_worker = EstimationWorker(self.api_key, self.translate_to, self.in_path, self.engine)
            file_map, all_strings = estimation_worker.process_files(files)

            strings_to_translate = [s for s in all_strings if not s.is_cached and s.needs_translation]
            total_items = len(strings_to_translate)

            self.log.emit(f"Starting translation of {total_items} strings...")

            for i in range(0, len(strings_to_translate), 10):
                if self.stop_requested:
                    self.log.emit("Translation stopped by user.")
                    return

                batch = strings_to_translate[i:i + 10]
                try:
                    batch_translate(batch, self.engine, self.translate_to)
                except Exception as e:
                    self.log.emit(f"Error in batch translation: {str(e)}")
                    continue

                progress = min((i + 10) / total_items * 100, 100)
                self.progress.emit(int(progress))
                self.log.emit(f"Translated {min(i + 10, total_items)}/{total_items} strings.")

            self.log.emit("Writing translated files...")
            self.write_translated_files(file_map)

            self.finished.emit()
        except Exception as e:
            self.error.emit(f"Translation error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}")

    def write_translated_files(self, file_map):
        for file, translation_file in file_map.items():
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    text_lines = f.readlines()

                for block in translation_file.translation_blocks:
                    for item in block.translation_items:
                        m = re.match(r'(.*)(".*")', text_lines[item.target_line])
                        if m:
                            text_lines[item.target_line] = f'{m.group(1)}{json.dumps(item.get_translated_content())}{item.suffix}\n'

                common_path = os.path.commonpath([os.path.abspath(file), os.path.abspath(self.in_path)])
                out_path_file = os.path.join(self.out_path, os.path.relpath(file, common_path))
                os.makedirs(os.path.dirname(out_path_file), exist_ok=True)

                with open(out_path_file, 'w', encoding='utf-8') as f:
                    f.writelines(text_lines)

                self.log.emit(f"Wrote translated file: {out_path_file}")
            except Exception as e:
                self.log.emit(f"Error writing file {file}: {str(e)}")

    def stop(self):
        self.stop_requested = True


class EstimationWorker(QThread):
    finished = pyqtSignal(int, float)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, api_key, translate_to, in_path, engine):
        super().__init__()
        self.api_key = api_key
        self.translate_to = translate_to
        self.in_path = in_path
        self.engine = engine
        self.stop_requested = False

    def run(self):
        try:
            self.progress.emit("Setting up API key...")
            openai.api_key = self.api_key

            self.progress.emit("Searching for files...")
            files = glob(os.path.join(self.in_path, "**", "*.rpy"), recursive=True)
            self.progress.emit(f"Found {len(files)} files to process.")

            self.progress.emit("Starting to process files...")
            file_map, all_strings = self.process_files(files)
            self.progress.emit(f"Processed {len(all_strings)} strings for translation.")

            self.progress.emit("Estimating tokens and cost...")
            total_tokens, estimated_cost = self.estimate_total_tokens(all_strings)

            self.finished.emit(total_tokens, estimated_cost)
        except Exception as e:
            error_msg = f"Estimation error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            self.error.emit(error_msg)

    def process_files(self, files: List[str]) -> Tuple[Dict[str, TranslationFile], List[TranslationString]]:
        file_map = {}
        all_strings = []

        for file in files:
            try:
                translation_file, strings = self.process_single_file(file)
                file_map[file] = translation_file
                all_strings.extend(strings)
                self.progress.emit(f"Processed file: {file}. Total strings so far: {len(all_strings)}")
            except Exception as e:
                self.progress.emit(f"Error processing file {file}: {str(e)}")

            if self.stop_requested:
                break

        return file_map, all_strings

    def process_single_file(self, file: str) -> Tuple[TranslationFile, List[TranslationString]]:
        try:
            translation_file = parse_file(file)
            strings = [item.translation_string for block in translation_file for item in block]

            for string in strings:
                if self.stop_requested:
                    break
                try:
                    string.translate(self.engine, self.translate_to)
                except Exception as e:
                    self.progress.emit(f"Error translating string in file {file}: {str(e)}")

            return translation_file, strings
        except Exception as e:
            tb = traceback.format_exc()
            print(tb)
            self.progress.emit(f"Error processing file {file}: {str(e)}")
            return TranslationFile(file), []

    def estimate_total_tokens(self, strings: List[TranslationString]) -> Tuple[int, Decimal]:
        total_tokens = 0
        uncached_strings = [s for s in strings if not s.is_cached and s.needs_translation]

        if not uncached_strings:
            return 0, Decimal('0')

        for i in range(0, len(uncached_strings), _BATCH_SIZE):
            batch = uncached_strings[i:i + _BATCH_SIZE]
            prefix = f'Translate the following text blocks to {self.translate_to}. Keep all markers and do not add new comments. Separate each translated block with "---". Here are the text blocks:'
            content = "\n".join([f"[{j + 1}] {item.content}" for j, item in enumerate(batch)])
            full_prompt = f"{prefix}\n\n{content}"

            total_tokens += estimate_tokens(self.engine, full_prompt) * 2

        self.progress.emit(f"Estimated tokens for {len(uncached_strings)} strings")
        estimated_cost = estimate_cost(self.engine, total_tokens)
        return total_tokens, estimated_cost

    def stop(self):
        self.stop_requested = True


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.estimation_worker = None
        self.setWindowTitle("Translation GUI")
        self.setGeometry(100, 100, 600, 400)

        # Initialize QSettings
        self.settings = QSettings("Akiba", "Ren'Py Translator")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # API Key
        api_key_layout = QHBoxLayout()
        api_key_layout.addWidget(QLabel("API Key:"))
        self.api_key_input = QLineEdit()
        api_key_layout.addWidget(self.api_key_input)
        layout.addLayout(api_key_layout)

        # Translate To
        translate_to_layout = QHBoxLayout()
        translate_to_layout.addWidget(QLabel("Translate To:"))
        self.translate_to_input = QLineEdit()
        translate_to_layout.addWidget(self.translate_to_input)
        layout.addLayout(translate_to_layout)

        # Input Path
        in_path_layout = QHBoxLayout()
        in_path_layout.addWidget(QLabel("Input Path:"))
        self.in_path_input = QLineEdit()
        in_path_layout.addWidget(self.in_path_input)
        self.in_path_button = QPushButton("Browse")
        self.in_path_button.clicked.connect(self.browse_in_path)
        in_path_layout.addWidget(self.in_path_button)
        layout.addLayout(in_path_layout)

        # Output Path
        out_path_layout = QHBoxLayout()
        out_path_layout.addWidget(QLabel("Output Path:"))
        self.out_path_input = QLineEdit()
        out_path_layout.addWidget(self.out_path_input)
        self.out_path_button = QPushButton("Browse")
        self.out_path_button.clicked.connect(self.browse_out_path)
        out_path_layout.addWidget(self.out_path_button)
        layout.addLayout(out_path_layout)

        # Engine Selection
        engine_layout = QHBoxLayout()
        engine_layout.addWidget(QLabel("Engine:"))
        self.engine_input = QComboBox()
        self.engine_input.addItems(_AVAILABLE_ENGINES)
        engine_layout.addWidget(self.engine_input)
        layout.addLayout(engine_layout)

        # Start Button
        self.start_button = QPushButton("Start Translation")
        self.start_button.clicked.connect(self.start_translation)
        layout.addWidget(self.start_button)

        # Add a Stop button
        self.stop_button = QPushButton("Stop Translation")
        self.stop_button.clicked.connect(self.stop_translation)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)

        # Progress Bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Log Output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)

        # Load saved settings
        self.load_settings()

        # Connect textChanged signals to save_settings
        self.api_key_input.textChanged.connect(self.save_settings)
        self.translate_to_input.textChanged.connect(self.save_settings)
        self.in_path_input.textChanged.connect(self.save_settings)
        self.out_path_input.textChanged.connect(self.save_settings)
        self.engine_input.currentTextChanged.connect(self.save_settings)

    def load_settings(self):
        self.api_key_input.setText(self.settings.value("api_key", ""))
        self.translate_to_input.setText(self.settings.value("translate_to", ""))
        self.in_path_input.setText(self.settings.value("in_path", ""))
        self.out_path_input.setText(self.settings.value("out_path", ""))
        saved_engine = self.settings.value("engine", _DEFAULT_ENGINE)
        index = self.engine_input.findText(saved_engine)
        if index >= 0:
            self.engine_input.setCurrentIndex(index)

    def save_settings(self):
        self.settings.setValue("api_key", self.api_key_input.text())
        self.settings.setValue("translate_to", self.translate_to_input.text())
        self.settings.setValue("in_path", self.in_path_input.text())
        self.settings.setValue("out_path", self.out_path_input.text())
        self.settings.setValue("engine", self.engine_input.currentText())

    def browse_in_path(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.in_path_input.setText(folder)

    def browse_out_path(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.out_path_input.setText(folder)

    def start_translation(self):
        api_key = self.api_key_input.text()
        translate_to = self.translate_to_input.text()
        in_path = self.in_path_input.text()
        out_path = self.out_path_input.text()
        engine = self.engine_input.currentText()

        if not all([api_key, translate_to, in_path, out_path, engine]):
            QMessageBox.warning(self, "Input Error", "Please fill in all fields.")
            return

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_output.clear()

        self.estimation_worker = EstimationWorker(api_key, translate_to, in_path, engine)
        self.estimation_worker.finished.connect(self.show_cost_confirmation)
        self.estimation_worker.error.connect(self.estimation_error)
        self.estimation_worker.progress.connect(self.log_output.append)
        self.estimation_worker.start()

        self.log_output.append("Estimating translation cost...")

    def stop_translation(self):
        if hasattr(self, 'worker'):
            self.worker.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log_output.append("Stopping translation...")

    def show_cost_confirmation(self, total_tokens, estimated_cost):
        reply = QMessageBox.question(self, 'Cost Estimation',
                                     f"Estimated total tokens: {total_tokens}\n"
                                     f"Estimated cost: ${estimated_cost:.2f}\n\n"
                                     "Do you want to proceed with the translation?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.proceed_translation()
        else:
            self.start_button.setEnabled(True)
            self.log_output.append("Translation cancelled by user.")

    def estimation_error(self, error_message):
        self.start_button.setEnabled(True)
        self.log_output.append(f"Estimation Error: {error_message}")
        QMessageBox.critical(self, "Estimation Error", f"An error occurred during cost estimation:\n\n{error_message}")

    def stop_estimation(self):
        if hasattr(self, 'estimation_worker'):
            self.estimation_worker.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log_output.append("Stopping estimation...")

    def proceed_translation(self):
        api_key = self.api_key_input.text()
        translate_to = self.translate_to_input.text()
        in_path = self.in_path_input.text()
        out_path = self.out_path_input.text()
        engine = self.engine_input.currentText()

        self.worker = TranslationWorker(api_key, translate_to, in_path, out_path, engine)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.translation_finished)
        self.worker.error.connect(self.translation_error)
        self.worker.start()

        self.log_output.append("Starting translation...")
        self.worker.log.connect(self.log_output.append)

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        self.log_output.append(f"Progress: {value}%")

    def translation_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log_output.append("Translation completed successfully!")
        QMessageBox.information(self, "Success", "Translation completed successfully!")

    def translation_error(self, error_message):
        self.start_button.setEnabled(True)
        self.log_output.append(f"Error: {error_message}")
        QMessageBox.critical(self, "Error", f"An error occurred: {error_message}")


def main():
    # Caching setup
    if not os.path.isfile(_TRANSLATION_CACHE_FILE):
        with open(_TRANSLATION_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump({}, f)

    with open(_TRANSLATION_CACHE_FILE, 'r', encoding='utf-8') as f:
        globals()['_TRANSLATION_CACHE'] = json.load(f)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
