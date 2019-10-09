SHELL := /bin/bash # Use bash syntax
PATH_DIR:=$(shell which conda | xargs -r dirname | xargs -r dirname )
PATH:="$(PATH_DIR)/:$(PATH_DIR)/Scripts/:$(PATH)"
VENV_ACTIVATE_FILE = ./venv
DIR_REPORTS = ./src

tmp:
	echo $(VERSION)

clean: ## Clean all data into target folder
	rm -Rf target || true

run: ## Run ASK Mail WS backend
	source activate ./venv; cd ./src/agent; python ./AgentBeacon.py

stop: info ## Stop ASK Mail backend
	ps -aef | grep 'python' | grep -v grep | awk '{print $$2}' | xargs -r kill -9

list: info ## List python version and venv modules
	source activate ./venv; conda env export; python --version

venv:
	conda remove -y -p ./venv --all || true
	rm -Rf src/venv || true
	conda env create -p ./venv -f src/environment.yml
	source activate ./venv; pip install pysc2

update-venv: ## Update venv (run venv once before running this option)
	conda env update -p ./venv -f src/environment.yml