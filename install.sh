#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

PLATFORM=""
PKG_MANAGER=""
APT_UPDATED=0

required_status=()
optional_status=()
next_steps=()

log_section() {
  printf '\n== %s ==\n' "$1"
}

add_required_status() {
  required_status+=("$1")
}

add_optional_status() {
  optional_status+=("$1")
}

add_next_step() {
  next_steps+=("$1")
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

is_tty() {
  [[ -t 0 ]]
}

detect_platform() {
  case "$(uname -s)" in
    Darwin)
      PLATFORM="macos"
      if command_exists brew; then
        PKG_MANAGER="brew"
      fi
      ;;
    Linux)
      if [[ -r /etc/os-release ]]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        case "${ID:-}" in
          ubuntu|debian)
            PLATFORM="debian"
            if command_exists apt-get; then
              PKG_MANAGER="apt"
            fi
            ;;
        esac
      fi
      ;;
  esac
}

ensure_supported_platform() {
  detect_platform

  if [[ "$PLATFORM" == "macos" && "$PKG_MANAGER" != "brew" ]]; then
    add_required_status "missing Homebrew on macOS"
    add_next_step "Install Homebrew from https://brew.sh, then re-run ./install.sh."
    return 1
  fi

  if [[ "$PLATFORM" == "debian" && "$PKG_MANAGER" != "apt" ]]; then
    add_required_status "missing apt on Ubuntu/Debian"
    add_next_step "Install required tools manually: uv, ffmpeg, then run uv sync."
    return 1
  fi

  if [[ -z "$PLATFORM" || -z "$PKG_MANAGER" ]]; then
    add_required_status "unsupported platform or package manager"
    add_next_step "Install uv and ffmpeg manually, then run uv sync."
    add_next_step "This installer currently supports macOS with Homebrew and Ubuntu/Debian with apt."
    return 1
  fi
}

run_apt_update_once() {
  if [[ "$APT_UPDATED" -eq 0 ]]; then
    sudo apt-get update
    APT_UPDATED=1
  fi
}

install_with_brew() {
  brew install "$1"
}

install_with_apt() {
  run_apt_update_once
  sudo apt-get install -y "$@"
}

install_uv() {
  case "$PKG_MANAGER" in
    brew)
      install_with_brew uv
      ;;
    apt)
      if ! command_exists curl; then
        install_with_apt curl ca-certificates
      fi
      curl -LsSf https://astral.sh/uv/install.sh | sh
      if [[ -d "$HOME/.local/bin" ]]; then
        export PATH="$HOME/.local/bin:$PATH"
      fi
      ;;
  esac
}

install_ffmpeg() {
  case "$PKG_MANAGER" in
    brew)
      install_with_brew ffmpeg
      ;;
    apt)
      install_with_apt ffmpeg
      ;;
  esac
}

install_ollama() {
  case "$PKG_MANAGER" in
    brew)
      install_with_brew ollama
      ;;
    apt)
      if ! command_exists curl; then
        install_with_apt curl ca-certificates
      fi
      curl -fsSL https://ollama.com/install.sh | sh
      ;;
  esac
}

ensure_required_dependency() {
  local cmd="$1"
  local label="$2"
  local installer="$3"

  if command_exists "$cmd"; then
    add_required_status "$label already installed"
    return 0
  fi

  printf 'Installing %s...\n' "$label"
  if "$installer"; then
    if command_exists "$cmd"; then
      add_required_status "$label installed"
      return 0
    fi
  fi

  add_required_status "$label installation failed"
  add_next_step "Install $label manually, then re-run ./install.sh."
  return 1
}

handle_optional_ollama() {
  if command_exists ollama; then
    add_optional_status "ollama already installed"
    return 0
  fi

  if is_tty; then
    local answer
    read -r -p "Ollama is optional and only needed for --summarize. Install it now? [y/N] " answer
    case "$answer" in
      [Yy]|[Yy][Ee][Ss])
        printf 'Installing ollama...\n'
        if install_ollama && command_exists ollama; then
          add_optional_status "ollama installed"
          add_next_step "Pull a local LLM before summarizing, for example: ollama pull llama3.2"
        else
          add_optional_status "ollama installation failed; summarization remains unavailable"
          add_next_step "Install Ollama manually later if you want --summarize support."
        fi
        return 0
        ;;
    esac
    add_optional_status "ollama skipped"
    add_next_step "Install Ollama later if you want --summarize support."
    return 0
  fi

  add_optional_status "ollama not installed"
  add_next_step "Install Ollama later if you want --summarize support."
  return 0
}

check_hf_token() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    printf 'HF_TOKEN is set in the current environment.\n'
  else
    printf 'HF_TOKEN is not set.\n'
    add_next_step "Accept the model terms at https://huggingface.co/pyannote/speaker-diarization-community-1 and https://huggingface.co/pyannote/segmentation-3.0."
    add_next_step "Export a token before running the script: export HF_TOKEN=your_token_here"
  fi
}

sync_python_environment() {
  printf 'Running uv sync...\n'
  (
    cd "$REPO_ROOT"
    uv sync
  )
}

print_status_list() {
  local heading="$1"
  shift
  local -a items=("$@")

  log_section "$heading"
  if [[ "${#items[@]}" -eq 0 ]]; then
    printf -- '- none\n'
    return
  fi

  local item
  for item in "${items[@]}"; do
    printf -- '- %s\n' "$item"
  done
}

main() {
  local setup_failed=0

  if ! ensure_supported_platform; then
    setup_failed=1
  fi

  if [[ "$setup_failed" -eq 0 ]]; then
    if ! ensure_required_dependency uv "uv" install_uv; then
      setup_failed=1
    fi

    if ! ensure_required_dependency ffmpeg "ffmpeg" install_ffmpeg; then
      setup_failed=1
    fi
  fi

  local python_env_status="skipped because required dependencies are missing"
  if [[ "$setup_failed" -eq 0 ]]; then
    if sync_python_environment; then
      python_env_status="uv environment ready"
    else
      python_env_status="uv sync failed"
      add_next_step "Resolve the uv sync failure above, then re-run ./install.sh."
      setup_failed=1
    fi
  fi

  if [[ "$setup_failed" -eq 0 ]]; then
    handle_optional_ollama
  else
    add_optional_status "ollama check skipped until required setup succeeds"
  fi

  print_status_list "Required dependencies" "${required_status[@]}"
  print_status_list "Optional dependencies" "${optional_status[@]}"

  log_section "Python environment"
  printf -- '- %s\n' "$python_env_status"

  log_section "Hugging Face token"
  check_hf_token

  print_status_list "Next steps" "${next_steps[@]}"

  if [[ "$setup_failed" -ne 0 ]]; then
    exit 1
  fi
}

main "$@"
