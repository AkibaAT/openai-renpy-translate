# ren'py translate - OpenAI | ChatGPT | Translate ren'py with Python

OpenAI ren'py translate is a simple command that can be used from the terminal to translate ren'py localization files.  
It uses OpenAI API to perform all operations. You can choose which model to use by passing it as a parameter, with
the current default being `gpt-3.5-turbo`.

If you would like to contribute, open a pull request, or issue, on GitHub.

### ren'py Fork Maintained By

- **[Akiba](https://github.com/AkibaAT)**

### Usage

#### Required

- Python >= 3.7
- **PAID** (or trial) OpenAI Account for official general API use with [API key](https://platform.openai.com/account/api-keys)  

### User Mode

1. Download the latest release for
[Windows](https://github.com/AkibaAT/openai-renpy-translate/releases/latest/download/openai-renpy-translate-windows-x86_64.exe)
or [Linux](https://github.com/AkibaAT/openai-renpy-translate/releases/latest/download/openai-renpy-translate-linux-x86_64)
2. Get your OpenAI [API key](https://platform.openai.com/account/api-keys)
3. Run the program:  
   1. Windows:  
   `openai-renpy-translate-windows-x86_64.exe`
   2. Linux:  
   `./openai-renpy-translate-linux-x86_64`

### Dev Mode

1. Setup env

```
git clone -b chatgpt https://github.com/AkibaAT/openai-renpy-translate.git
cd openai-renpy-translate
./scripts/setup.sh
source .activate
```

2. Usage

`python src/app.py --api-key ${OPENAI_API_KEY} --translate ${TARGET_LANGUAGE} 
--in-path "renpyProject/game/tl/${TARGET_LANGUAGE}" --out-path "renpyProject/game/tl/${TARGET_LANGUAGE}-AI"`

### Contributors

<a href="https://github.com/AkibaAT/openai-renpy-translate/graphs/contributors"> <img src="https://contrib.rocks/image?repo=AkibaAT/openai-renpy-translate" /> </a>

### License

The project is made available under the GPL-3.0 license. See the `LICENSE` file for more information.
