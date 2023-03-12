# Opentranslator - OpenAI | ChatGPT | Translate ren'py with python

Opentranslator is a simple command that can be used from the terminal to translate ren'py localization files.  
It uses OpenAI API to perform all operations, you can choose which model to use by passing it as a parameter.

The following project was born to study OpenAI and how it works.
If you would contribute open a pull request or issue on GitHub.

### ren'py Fork Maintained By

- **[Rejaku](https://github.com/Rejaku)**

### Usage

#### Required

- Python >= 3.7
- OpenAI Account


### Dev Mode

1. Setup env

```
git clone -b chatgpt git@github.com:Rejaku/openai-chatgpt-opentranslator.git
cd chatgpt-translate-app
echo 'export OPENAI_API_KEY="${YOUR_OPENAI_API_KEY}"' > .env
chmod +x scritps/*.sh
./scripts/setup.sh
source .activate
```

2. Usage

```
python opentranslator/app.py --translate German --in-path renpyProject/game/tl/German --out-path renpyProject/game/tl/German-AI
```

### Contributors

<a href="https://github.com/Rejaku/openai-renpy-translate/graphs/contributors"> <img src="https://contrib.rocks/image?repo=Rejaku/openai-renpy-translate" /> </a>

### License

The project is made available under the GPL-3.0 license. See the `LICENSE` file for more information.