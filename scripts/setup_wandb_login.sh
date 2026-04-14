#!/usr/bin/env zsh
set -euo pipefail

# Securely configure W&B credentials on macOS/Linux without storing secrets in repo.
NETRC_PATH="$HOME/.netrc"
API_KEY_ARG="${1:-}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 no esta disponible en PATH"
  exit 1
fi

# Ensure wandb python package is available for the selected python3.
if ! python3 - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("wandb") is not None else 1)
PY
then
  echo "Instalando wandb para python3..."
  python3 -m pip install wandb
fi

# Priority: argument > existing environment > interactive prompt.
if [[ -n "$API_KEY_ARG" ]]; then
  WANDB_API_KEY="$API_KEY_ARG"
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  if [[ -t 0 ]]; then
    echo -n "Introduce tu WANDB API key (input oculto): "
    read -r -s WANDB_API_KEY
    echo
  else
    echo "No hay terminal interactiva para pedir la API key."
    echo "Usa una de estas opciones:"
    echo "  1) WANDB_API_KEY=tu_key ./scripts/setup_wandb_login.sh"
    echo "  2) ./scripts/setup_wandb_login.sh tu_key"
    exit 1
  fi
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "No se recibio ninguna API key."
  exit 1
fi

{
  echo "machine api.wandb.ai"
  echo "  login user"
  echo "  password ${WANDB_API_KEY}"
} > "$NETRC_PATH"

chmod 600 "$NETRC_PATH"

echo "Credenciales guardadas en $NETRC_PATH con permisos 600."

echo "Verificando login de W&B..."
python3 - <<'PY'
import wandb
wandb.login(relogin=False)
print("W&B login OK")
PY

echo "Listo. Puedes usar W&B sin volver a hacer login interactivo."
