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
import tiktoken
openai.api_key = os.environ['OPENAI_API_KEY']

_VERBOSITY = 1

_TRANSLATION_CACHE = {}
_TRANSLATION_CACHE_FILE = 'translation_cache.json'
_TEMPERATURE = 0.7
_MAX_TOKEN = 256
_UNOFFICIAL_MODE = False

# Link https://beta.openai.com/docs/models/gpt-3
_DEFAULT_ENGINE = 'gpt-3.5-turbo'
_AVAILABLE_ENGINES = [
    'text-curie-001',
    'text-babbage-001',
    'text-ada-001',
    'davinci',
    'curie',
    'babbage',
    'ada',
    'gpt-3.5-turbo',
    'text-davinci-003',
    'text-davinci-002',
    'gpt-4',
    'gpt-4-32k'
]

# Link https://openai.com/pricing#language-models
_PRICING_PER_1K_TOKENS = {
    'text-curie-001': 0.002,
    'text-babbage-001': 0.0005,
    'text-ada-001': 0.0004,
    'davinci': 0.02,
    'curie': 0.002,
    'babbage': 0.0005,
    'ada': 0.0004,
    'gpt-3.5-turbo': 0.002,
    'text-davinci-003': 0,
    'text-davinci-002': 0,
    'gpt-4': 0.06,
    'gpt-4-32k': 0.12
}

# Link https://platform.openai.com/docs/models/overview
_TOKEN_LIMITS_PER_REQUEST = {
    'text-curie-001': 2049,
    'text-babbage-001': 2049,
    'text-ada-001': 2049,
    'davinci': 2049,
    'curie': 2049,
    'babbage': 2049,
    'ada': 2049,
    'gpt-3.5-turbo': 4096,
    'text-davinci-003': 4096,
    'text-davinci-002': 4096,
    'gpt-4': 8192,
    'gpt-4-32k': 32768
}


def set_verbosity(verbosity: int):
    if 0 < verbosity <= 3:
        globals()['_VERBOSITY'] = verbosity


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
    print("Persisting Cache")
    with open(_TRANSLATION_CACHE_FILE, 'w', encoding='utf-8') as f:
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
        self.original_content = original_content
        self.suffix = suffix

    def translate(self, engine, to_language):
        self.translation_string.translate(engine, to_language)

    def get_translated_content(self):
        return repr(self.translation_string.translation)

    @property
    def original_content(self):
        return self._original_content

    @original_content.setter
    def original_content(self, original_content):
        self._original_content = original_content
        self.translation_string = TranslationString(self.original_content)

    def __repr__(self):
        return "TranslationItem({}, {})".format(self.source_line + 1, self.target_line + 1)


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

    def translate(self, engine, to_language):
        self.engine = engine
        self.to_language = to_language

        if self.translation:
            return

        if not self.content.strip():
            self.translation = self.content
            return

        cached_translation = self.pull_from_cache(to_language)
        if cached_translation:
            self.translation = cached_translation
            return

        prefix = f'Translate to {self.to_language}, keep all markers, don\'t add new comments:'
        text = f'{prefix}:\n{self.content}'

        request_token = calculate_tokens(engine, text)
        request_pricing = calculate_pricing(engine, request_token)
        debug(f'# Request cost: ${request_pricing}')
        debug(
            f'# Request token limit: {request_token}/{_TOKEN_LIMITS_PER_REQUEST[engine]}')

        if request_token > _TOKEN_LIMITS_PER_REQUEST[engine]:
            raise Exception(
                f'The required tokens {request_token} are greater than the limit of {_TOKEN_LIMITS_PER_REQUEST[engine]}')

        if _UNOFFICIAL_MODE:
            api_base = "https://chatgpt-api.shn.hk/v1"
        else:
            api_base = None
        response = None
        for i in range(3):
            try:
                if engine == 'gpt-3.5-turbo':
                    response = openai.ChatCompletion.create(
                        request_timeout=10,
                        api_base=api_base,
                        model=engine,
                        messages=[{"role": "user", "content": text}],
                        temperature=_TEMPERATURE,
                        max_tokens=_MAX_TOKEN,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
                    if choices := response.get('choices', []):
                        if len(choices) > 0:
                            self.translation = choices[0]['message']['content'].lstrip().replace('"', '\\"')
                            break
                else:
                    response = openai.Completion.create(
                        request_timeout=10,
                        api_base=api_base,
                        model=engine,
                        prompt=text,
                        temperature=_TEMPERATURE,
                        max_tokens=_MAX_TOKEN,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
                    if choices := response.get('choices', []):
                        if len(choices) > 0:
                            self.translation = choices[0]['text'].lstrip().replace('"', '\\"')
                            break
            except openai.error.Timeout:
                if i < 2:
                    debug('Request had error, retrying...')
                else:
                    debug('Request had error, skipping!')
                continue
        if response is not None:
            response_token = response.get('usage', {}).get('completion_tokens', 0)
            response_pricing = calculate_pricing(engine, response_token)
            debug(f'# Response cost: ${response_pricing}')
            debug(f'# Response tokens: {response_token}')
            debug(f'# Total tokens: {request_token + response_token}')
            debug(f'# Total cost: ${request_pricing + response_pricing}')
            _TRANSLATION_CACHE[self.content] = {to_language: self.translation}
            persist_cache()

    def pull_from_cache(self, to_language):
        available_translations = _TRANSLATION_CACHE.get(self.content, None)
        if available_translations:
            cached_translation = available_translations.get(to_language, None)
            if cached_translation:
                return cached_translation

    def __repr__(self):
        return 'TranslationString(content="{}", translation="{}")'.format(self.content, self.translation)


@click.command()
@click.option('--translate', type=str, required=True)
@click.option('--in-path', type=str, required=True, help='(required) File path containing the text')
@click.option('--out-path', type=str, required=True, help='(required) The directory to output data to')
@click.option('--engine', default=_DEFAULT_ENGINE, type=click.Choice(_AVAILABLE_ENGINES), help='GPT-3 engine')
@click.option('--unofficial', default=False, type=bool, help='Use the unofficial GPT-3 API')
@click.option('--temperature', default=_TEMPERATURE, help='Higher values means the model will take more risks. Values 0 to 1')
@click.option('--max-token', default=_MAX_TOKEN, help='The maximum number of tokens to generate in the completion.')
@click.option('-v', '--verbosity', default=3, count=True)
def main(translate, in_path, out_path, engine, unofficial, max_token, temperature, verbosity):
    try:
        set_verbosity(verbosity)

        globals()['_MAX_TOKEN'] = max_token
        globals()['_TEMPERATURE'] = temperature
        globals()['_UNOFFICIAL_MODE'] = unofficial

        # Caching setup
        if not os.path.isfile(_TRANSLATION_CACHE_FILE):
            with open(_TRANSLATION_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump({}, f)

        with open(_TRANSLATION_CACHE_FILE, 'r', encoding='utf-8') as f:
            globals()['_TRANSLATION_CACHE'] = json.load(f)

        # File Parsing
        files = glob(os.path.join(in_path, "**", "*.rpy"), recursive=True)
        file_map = {}

        # Find all translation blocks in all files
        # Create a mapping to the correct line in each file
        print("Parsing files")
        for file in tqdm(files, total=len(files), unit="files"):
            with open(file, 'r', encoding='utf-8') as f:
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
                        m = re.match(r'(\s*)(\w+)?\s*"(.*)"', line.strip())
                        if m is not None:
                            translation_item.translation_string.translation = m.group(3)
                        translation_item.target_line = i
                        translation_block.add_translation_item(translation_item)
                        continue

                    if text_lines[i - 1].strip().startswith("#") \
                            and not text_lines[i - 1].strip().startswith("# nvl clear") \
                            and line.strip():
                        source = re.match(r'(\s*)(\w+)?\s*"(.*)"', text_lines[i - 1].strip()[2:])
                        if source is None:
                            continue
                        position = text_lines[i - 1].find(source.group(3))
                        suffix = text_lines[i - 1][position + len(source.group(3)) + 1:]
                        if line.strip().startswith("nvl clear"):
                            translation_item = TranslationItem(source_line=i,
                                                               original_content=source.group(3),
                                                               suffix=suffix,
                                                               target_line=i + 1)
                        else:
                            translation_item = TranslationItem(source_line=i,
                                                               original_content=source.group(3),
                                                               suffix=suffix,
                                                               target_line=i)
                        translation = re.match(r'(\s*)(\w+)?\s*"(.*)"', text_lines[i].strip())
                        if translation is not None:
                            translation_item.translation_string.translation = translation.group(3)
                        translation_block.add_translation_item(translation_item)
                        continue

                if translation_block:
                    translation_file.add_translation_block(translation_block)
                file_map[file] = translation_file

                print('Starting translation')
                tqdm.write("Translating '{}' {}\u2713 OK{}".format(translation_file, Fore.GREEN, Style.RESET_ALL))
                translation_file.translate(engine, translate)

                # Write back to disk
                print("Saving translations")
                with open(file, 'r', encoding='utf-8') as f:
                    text_lines = f.readlines()
                    for block in tqdm(translation_file, total=len(translation_file.translation_blocks), unit="blocks"):
                        for item in block:
                            m = re.match(r"(\s*)(\w+)?\s*("")", text_lines[item.target_line])
                            if m is None:
                                continue
                            if m.group(2):
                                text_lines[item.target_line] = '{}{} "{}"{}\n'.format(m.group(1), m.group(2),
                                                                                      item.get_translated_content(),
                                                                                      item.suffix)
                            else:
                                text_lines[item.target_line] = '{}"{}"{}\n'.format(m.group(1),
                                                                                   item.get_translated_content(),
                                                                                   item.suffix)
                persist_cache()
                common_path = os.path.commonpath([os.path.abspath(file), os.path.abspath(in_path)])
                out_path_file = os.path.join(out_path, os.path.relpath(file, common_path))
                os.makedirs(os.path.dirname(out_path_file), exist_ok=True)
                with open(out_path_file, 'w', encoding='utf-8') as f:
                    f.writelines(text_lines)
                print("")
    except Exception as e:
        echo(f'[ERR] {str(e)}')
        sys.exit(1)


if __name__ == '__main__':
    sys.exit(main())
