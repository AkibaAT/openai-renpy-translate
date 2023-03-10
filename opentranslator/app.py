import os
import pathlib
import sys
import re
from glob import glob
from decimal import Decimal
from tqdm import tqdm
from colorama import Fore, Style

import json
import click
import openai

_VERBOSITY = 1

_TRANSLATION_CACHE_FILE = 'translation_cache.json'

# Calculate token https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them

# Link https://beta.openai.com/docs/models/gpt-3
_DEFAULT_ENGINE = 'gpt-3.5-turbo'
_AVAILABLE_ENGINES = ['text-ada-001', 'text-babbage-001',
                      'text-curie-001', 'text-davinci-003',
                      'gpt-3.5-turbo']

# Link https://openai.com/api/pricing
# 750 words = 1000 tokens
_WORDS = 750
_TOKEN = 1000
_PRICING_PER_1K_TOKENS = {
    'text-ada-001': 0.0004,
    'text-babbage-001': 0.0005,
    'text-curie-001': 0.0020,
    'text-davinci-003': 0.0200,
    'gpt-3.5-turbo': 0.0020
}

# Link https://beta.openai.com/docs/models/gpt-3
_TOKEN_LIMITS_PER_REQUEST = {
    'text-ada-001': 2048,
    'text-babbage-001': 2048,
    'text-curie-001': 2048,
    'text-davinci-003': 4000,
    'gpt-3.5-turbo': 4000
}


def set_verbosity(verbose: int):
    if verbose > 0 and verbose <= 3:
        globals()['_VERBOSITY'] = verbose


def render_file(filepath: str) -> str:
    with open(str(pathlib.Path(filepath)), 'r') as f:
        return f.read()


def calculate_pricing(engine: str, tokens: int) -> Decimal:
    pricing = Decimal((tokens / _TOKEN) * _PRICING_PER_1K_TOKENS[engine])
    return round(pricing, 4)


def verbose(level: str):
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


class TranslationItem():

    def __init__(self, source_line=0, target_line=0, original_content="", translated_content=""):
        self.source_line = source_line
        self.target_line = target_line
        self.translation_string = None
        self.original_content = original_content

    def translate(self, engine, to_language):
        self.translation_string.translate(engine, to_language)

    def get_translated_content(self):
        return self.translation_string.translation

    @property
    def original_content(self):
        return self._original_content

    @original_content.setter
    def original_content(self, original_content):
        self._original_content = original_content
        self.translation_string = TranslationString(self.original_content)

    def __repr__(self):
        return "TranslationItem({}, {})".format(self.source_line + 1, self.target_line + 1)


class TranslationBlock():

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

class TranslationString():

    def __init__(self, content):
        self.content = content
        self.engine = None
        self.to_language = None
        self.translation = None

    def translate(self, engine, to_language):
        self.engine = engine
        self.to_language = to_language

        if not self.content.strip():
            self.translation = self.content
            return

        cached_translation = self.pull_from_cache(to_language)
        if cached_translation:
            self.translation = cached_translation
            return

        prefix = f'Translate the following text into {self.to_language}'
        text = f'{prefix}:\n{self.content}'

        request_token = round(_WORDS / len(text))
        request_pricing = calculate_pricing(engine, request_token)
        debug(f'# Request cost: ${request_pricing}')
        debug(
            f'# Request token limit: {request_token}/{_TOKEN_LIMITS_PER_REQUEST[engine]}')

        if request_token > _TOKEN_LIMITS_PER_REQUEST[engine]:
            raise Exception(
                f'The required tokens {request_token} are greater than the limit of {_TOKEN_LIMITS_PER_REQUEST[engine]}')

        if engine == 'gpt-3.5-turbo':
            response = openai.ChatCompletion.create(
                model=engine,
                messages=[{"role": "user", "content": text}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKEN,
                frequency_penalty=0,
                presence_penalty=0
            )
            if choices := response.get('choices', []):
                if len(choices) > 0:
                    self.translation = choices[0]['message']['content'].lstrip().replace('"', '\\"')
        else:
            response = openai.Completion.create(
                model=engine,
                prompt=text,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKEN,
                frequency_penalty=0,
                presence_penalty=0
            )
            if choices := response.get('choices', []):
                if len(choices) > 0:
                    self.translation = choices[0]['text'].lstrip().replace('"', '\\"')
        response_token = response.get('usage', {}).get('completion_tokens', 0)
        response_pricing = calculate_pricing(engine, response_token)
        debug(f'# Response cost: ${response_pricing}')
        debug(f'# Response tokens: {response_token}')
        debug(f'# Total tokens: {request_token + response_token}')
        debug(f'# Total cost: ${request_pricing + response_pricing}')
        TRANSLATION_CACHE[self.content] = {to_language: self.translation}

    def pull_from_cache(self, to_language):
        available_translations = TRANSLATION_CACHE.get(self.content, None)
        if available_translations:
            cached_translation = available_translations.get(to_language, None)
            if cached_translation:
                return cached_translation

    def __repr__(self):
        return 'TranslationString(content="{}", translation="{}")'.format(self.content, self.translation)

@click.command()
@click.option('--translate', type=str, required=True)
@click.option('--text', default=None, help='Raw text')
@click.option('--filepath', type=str, required=True, help='File path containing the text')
@click.option('--output', type=str, required=True, help="(required) The directory to output data to")
@click.option('--engine', default=_DEFAULT_ENGINE, type=click.Choice(_AVAILABLE_ENGINES), help='GPT-3 engine')
@click.option('--temperature', default=0.7, help='Higher values means the model will take more risks. Values 0 to 1')
@click.option('--max-token', default=256, help='The maximum number of tokens to generate in the completion.')
@click.option('-v', '--verbose', default=3, count=True)
def main(translate, text, filepath, output, engine, max_token, temperature, verbose):
    try:
        set_verbosity(verbose)

        global TRANSLATION_CACHE
        global MAX_TOKEN
        global TEMPERATURE
        MAX_TOKEN = max_token
        TEMPERATURE = temperature

        # Caching setup
        if not os.path.isfile(_TRANSLATION_CACHE_FILE):
            with open(_TRANSLATION_CACHE_FILE, "w") as f:
                json.dump({}, f)

        with open(_TRANSLATION_CACHE_FILE, "r") as f:
            TRANSLATION_CACHE = json.load(f)

        # File Parsing
        files = glob(os.path.join(filepath, "**", "*.rpy"), recursive=True)
        file_map = {}

        # Find all translation blocks in all files
        # Create a mapping to the correct line in each file
        print("Parsing files")
        for file in tqdm(files, total=len(files), unit="files"):
            with open(file, "r") as f:
                translation_file = TranslationFile(file)
                translation_block, translation_item, block = None, None, None
                text_lines = f.readlines()
                for i, line in enumerate(text_lines):
                    if i != block and re.match(r"translate ([A-z0-9_]+) ([A-z0-9_]+)", line):
                        block = i
                        if translation_block:
                            translation_file.add_translation_block(translation_block)
                        translation_block = TranslationBlock(source_file=file, block_line=i)
                        continue

                    if text_lines[i - 1].strip().startswith("#") and line.strip().startswith("old"):
                        m = re.match(r'(\s*)(\w+)?\s*"(.*)"', line.strip())
                        translation_item = TranslationItem(source_line=i, original_content=m.group(3))
                        continue

                    if text_lines[i - 1].strip().startswith("old") and line.strip().startswith("new"):
                        translation_item.target_line = i
                        translation_block.add_translation_item(translation_item)
                        continue

                    if text_lines[i - 1].strip().startswith("#") \
                            and not text_lines[i - 1].strip().startswith("# nvl clear") \
                            and line.strip():
                        m = re.match(r'(\s*)(\w+)?\s*"(.*)"', text_lines[i - 1].strip()[2:])
                        if line.strip().startswith("nvl clear"):
                            translation_item = TranslationItem(source_line=i,
                                                               original_content=m.group(3),
                                                               target_line=i + 1)
                        else:
                            translation_item = TranslationItem(source_line=i,
                                                               original_content=m.group(3),
                                                               target_line=i)
                        translation_block.add_translation_item(translation_item)
                        continue

                if translation_block:
                    translation_file.add_translation_block(translation_block)
                file_map[file] = translation_file

        print('Starting translation')
        for file, translation_file in tqdm(file_map.items(), total=len(file_map.keys()), unit="files"):
            tqdm.write("Translating '{}' {}\u2713 OK{}".format(translation_file, Fore.GREEN, Style.RESET_ALL))
            translation_file.translate(engine, translate)

        # Write back to disk
        print("Saving translations")
        for file, translation_file in tqdm(file_map.items(), total=len(file_map.keys()), unit="files"):
            with open(file, "r") as f:
                text_lines = f.readlines()
                for block in tqdm(translation_file, total=len(translation_file.translation_blocks), unit="blocks"):
                    for item in block:
                        m = re.match(r"(\s*)(\w+)?\s*("")", text_lines[item.target_line])
                        if m.group(2):
                            text_lines[item.target_line] = '{}{} "{}"\n'.format(m.group(1), m.group(2),
                                                                                item.get_translated_content())
                        else:
                            text_lines[item.target_line] = '{}"{}"\n'.format(m.group(1), item.get_translated_content())
            common_path = os.path.commonpath([os.path.abspath(file), os.path.abspath(filepath)])
            out_path = os.path.join(output, os.path.relpath(file, common_path))
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                f.writelines(text_lines)
        print("")

        print("Persisting Cache")
        with open(_TRANSLATION_CACHE_FILE, "w") as f:
            json.dump(TRANSLATION_CACHE, f, indent=4)
    except Exception as e:
        echo(f'[ERR] {str(e)}')
        sys.exit(1)


if __name__ == '__main__':
    sys.exit(main())