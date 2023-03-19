# Opentranslator - OpenAI | ChatGPT | Translate ren'py with python

Opentranslator is a simple command that can be used from the terminal to translate ren'py localization files.  
It uses OpenAI API to perform all operations, you can choose which model to use by passing it as a parameter.

If you would like to contribute, open a pull request, or issue, on GitHub.

### ren'py Fork Maintained By

- **[Rejaku](https://github.com/Rejaku)**

### Usage

#### Required

- Python >= 3.7
- **PAID** OpenAI Account (unless you use --unofficial 1)

### User Mode

1. Download the latest release for
[Windows](https://github.com/Rejaku/openai-renpy-translate/releases/latest/download/openai-renpy-translate-windows-x86_64.exe)
or [Linux](https://github.com/Rejaku/openai-renpy-translate/releases/latest/download/openai-renpy-translate-linux-x86_64)
2. Run the program:  
   1. Windows:  
   `set OPENAI_API_KEY=${YOUR_OPENAI_API_KEY}`  
   `openai-renpy-translate-windows-x86_64.exe --translate "${TARGET_LANGUAGE}" 
   --in-path renpyProject/game/tl/${TARGET_LANGUAGE} --out-path renpyProject/game/tl/${TARGET_LANGUAGE}-AI`
   2. Linux:  
   `export OPENAI_API_KEY=${YOUR_OPENAI_API_KEY}`  
   `./openai-renpy-translate-linux-x86_64 --translate "${TARGET_LANGUAGE}" 
   --in-path renpyProject/game/tl/${TARGET_LANGUAGE} --out-path renpyProject/game/tl/${TARGET_LANGUAGE}-AI`

### Dev Mode

1. Setup env

```
git clone -b chatgpt git@github.com:Rejaku/openai-chatgpt-opentranslator.git
cd chatgpt-translate-app
export OPENAI_API_KEY=${YOUR_OPENAI_API_KEY}
./scripts/setup.sh
source .activate
```

2. Usage

`python src/app.py --translate ${TARGET_LANGUAGE} --in-path renpyProject/game/tl/${TARGET_LANGUAGE} 
--out-path renpyProject/game/tl/${TARGET_LANGUAGE}-AI`

### Contributors

<a href="https://github.com/Rejaku/openai-renpy-translate/graphs/contributors"> <img src="https://contrib.rocks/image?repo=Rejaku/openai-renpy-translate" /> </a>

### License

The project is made available under the GPL-3.0 license. See the `LICENSE` file for more information.
