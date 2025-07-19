#!/bin/bash

set -euo pipefail

# --- Directory context management ---
CURRENT_DIR="$(pwd)"
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Script configuration
SCRIPT_NAME=$(basename "$0")
COMPOSE_FILE="docker-compose.yaml"
LOG_FILE="/tmp/${SCRIPT_NAME%.*}.log"

# Colors for output (only if stdout is a terminal)
if [ -t 1 ]; then
    RED=$(printf '\033[0;31m')
    GREEN=$(printf '\033[0;32m')
    YELLOW=$(printf '\033[1;33m')
    BLUE=$(printf '\033[0;34m')
    NC=$(printf '\033[0m')
else
    RED=""
    GREEN=""
    YELLOW=""
    BLUE=""
    NC=""
fi

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

# Colored output functions with emoticons
info() {
    printf "%bℹ️  [INFO]%b %s\n" "$BLUE" "$NC" "$1"
    log "INFO: $1"
}

success() {
    printf "%b✅ [SUCCESS]%b %s\n" "$GREEN" "$NC" "$1"
    log "SUCCESS: $1"
}

warning() {
    printf "%b⚠️  [WARNING]%b %s\n" "$YELLOW" "$NC" "$1"
    log "WARNING: $1"
}

error() {
    printf "%b❌ [ERROR]%b %s\n" "$RED" "$NC" "$1" >&2
    log "ERROR: $1"
}

show_help() {
    # Use printf for color codes to avoid literal escape output
    printf "🐳 %bDocker Compose Management Script%b\n\n" "$BLUE" "$NC"

    printf "%b📋 Usage:%b %s [OPTIONS]\n\n" "$YELLOW" "$NC" "$SCRIPT_NAME"

    printf "%b⚙️  Options:%b\n" "$YELLOW" "$NC"
    printf "  --build              🔨 Build the container image (with cache)\n"
    printf "  --clean-build        🧹 Build the container image with no cache\n"
    printf "  --start              🚀 Start the container in detached mode\n"
    printf "  --stop               🛑 Stop and remove containers\n"
    printf "  --restart            🔄 Restart the containers\n"
    printf "  --logs [SERVICE]     📄 Show logs (optionally for specific service)\n"
    printf "  --status             📊 Show container status\n"
    printf "  --clean              🧹 Remove containers, networks, and volumes\n"
    printf "  --pull               📥 Pull latest images before building\n"
    printf "  --file FILE          📁 Use specific compose file (default: %s)\n" "$COMPOSE_FILE"
    printf "  --verbose            🔊 Enable verbose output\n"
    printf "  --help               ❓ Show this help message and exit\n\n"

    printf "%b💡 Examples:%b\n" "$YELLOW" "$NC"
    printf "  %s --build --start        # 🔨🚀 Build (cached) and start containers\n" "$SCRIPT_NAME"
    printf "  %s --clean-build --start  # 🧹🔨🚀 Clean build and start containers\n" "$SCRIPT_NAME"
    printf "  %s --restart              # 🔄 Restart all containers\n" "$SCRIPT_NAME"
    printf "  %s --logs web             # 📄 Show logs for 'web' service\n" "$SCRIPT_NAME"
    printf "  %s --status               # 📊 Check container status\n" "$SCRIPT_NAME"
    printf "  %s --clean                # 🧹 Clean up everything\n" "$SCRIPT_NAME"
    printf "  %s --pull --clean-build   # 📥🧹🔨 Pull images and clean rebuild\n\n" "$SCRIPT_NAME"

    printf "%b🛠️  Additional Commands:%b\n" "$YELLOW" "$NC"
    printf "  docker compose down             # 🛑 Stop containers (manual)\n"
    printf "  docker compose logs -f          # 📄 Follow logs (manual)\n\n"

    printf "%b📝 Log file:%b %s\n" "$YELLOW" "$NC" "$LOG_FILE"
}

# Check if Docker and Docker Compose are available
check_dependencies() {
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
        exit 1
    fi

    if ! docker compose version &> /dev/null; then
        error "Docker Compose is not available"
        exit 1
    fi
}

# Check if compose file exists
check_compose_file() {
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        error "Compose file '$COMPOSE_FILE' not found in current directory"
        exit 1
    fi
}

# Check if Docker daemon is running
check_docker_daemon() {
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        exit 1
    fi
}

# Build containers (with cache)
build_containers() {
    info "Building container images... 🔨"
    if [[ $VERBOSE -eq 1 ]]; then
        docker compose -f "$COMPOSE_FILE" build
    else
        docker compose -f "$COMPOSE_FILE" build > /dev/null 2>&1
    fi
    success "Container images built successfully"
}

# Build containers without cache
clean_build_containers() {
    info "Building container images with no cache... 🧹"
    if [[ $VERBOSE -eq 1 ]]; then
        docker compose -f "$COMPOSE_FILE" build --no-cache
    else
        docker compose -f "$COMPOSE_FILE" build --no-cache > /dev/null 2>&1
    fi
    success "Container images built successfully"
}

# Start containers
start_containers() {
    info "Starting containers in detached mode... 🚀"
    if [[ $VERBOSE -eq 1 ]]; then
        docker compose -f "$COMPOSE_FILE" up -d
    else
        docker compose -f "$COMPOSE_FILE" up -d > /dev/null 2>&1
    fi
    success "Containers started successfully"
}

# Stop containers
stop_containers() {
    info "Stopping containers... 🛑"
    if [[ $VERBOSE -eq 1 ]]; then
        docker compose -f "$COMPOSE_FILE" down
    else
        docker compose -f "$COMPOSE_FILE" down > /dev/null 2>&1
    fi
    success "Containers stopped successfully"
}

# Restart containers
restart_containers() {
    info "Restarting containers... 🔄"
    if [[ $VERBOSE -eq 1 ]]; then
        docker compose -f "$COMPOSE_FILE" restart
    else
        docker compose -f "$COMPOSE_FILE" restart > /dev/null 2>&1
    fi
    success "Containers restarted successfully"
}

# Show logs
show_logs() {
    local service="${1:-}"
    if [[ -n "$service" ]]; then
        info "Showing logs for service: $service 📄"
        docker compose -f "$COMPOSE_FILE" logs -f "$service"
    else
        info "Showing logs for all services 📄"
        docker compose -f "$COMPOSE_FILE" logs -f
    fi
}

# Show container status
show_status() {
    info "Container status: 📊"
    docker compose -f "$COMPOSE_FILE" ps
    echo
    info "Resource usage: 💾"
    docker compose -f "$COMPOSE_FILE" top 2>/dev/null || warning "Unable to show resource usage"
}

# Clean up everything
clean_containers() {
    warning "This will remove all containers, networks, and volumes! 🧹💥"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        info "Cleaning up containers, networks, and volumes... 🧹"
        docker compose -f "$COMPOSE_FILE" down -v --remove-orphans
        success "Cleanup completed"
    else
        info "Cleanup cancelled 🚫"
    fi
}

# Pull latest images
pull_images() {
    info "Pulling latest images... 📥"
    if [[ $VERBOSE -eq 1 ]]; then
        docker compose -f "$COMPOSE_FILE" pull
    else
        docker compose -f "$COMPOSE_FILE" pull > /dev/null 2>&1
    fi
    success "Images pulled successfully"
}

# Initialize variables
BUILD=0
CLEAN_BUILD=0
START=0
STOP=0
RESTART=0
LOGS=0
STATUS=0
CLEAN=0
PULL=0
VERBOSE=0
LOG_SERVICE=""

# Handle no arguments
if [[ $# -eq 0 ]]; then
    show_help
    exit 1
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build)
            BUILD=1
            shift
            ;;
        --clean-build)
            CLEAN_BUILD=1
            shift
            ;;
        --start)
            START=1
            shift
            ;;
        --stop)
            STOP=1
            shift
            ;;
        --restart)
            RESTART=1
            shift
            ;;
        --logs)
            LOGS=1
            if [[ $# -gt 1 && ! "$2" =~ ^-- ]]; then
                LOG_SERVICE="$2"
                shift
            fi
            shift
            ;;
        --status)
            STATUS=1
            shift
            ;;
        --clean)
            CLEAN=1
            shift
            ;;
        --pull)
            PULL=1
            shift
            ;;
        --file)
            if [[ $# -lt 2 ]]; then
                error "--file requires a filename argument"
                exit 1
            fi
            COMPOSE_FILE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    # Change to repo root before doing any work
    cd "$REPO_ROOT"

    # Perform checks
    check_dependencies
    check_docker_daemon
    check_compose_file

    info "Using compose file: $COMPOSE_FILE 📁"

    # Execute operations in logical order
    if [[ $CLEAN -eq 1 ]]; then
        clean_containers
        cd "$CURRENT_DIR"
        exit 0
    fi

    if [[ $PULL -eq 1 ]]; then
        pull_images
    fi

    if [[ $BUILD -eq 1 ]]; then
        build_containers
    fi

    if [[ $CLEAN_BUILD -eq 1 ]]; then
        clean_build_containers
    fi

    if [[ $STOP -eq 1 ]]; then
        stop_containers
    fi

    if [[ $START -eq 1 ]]; then
        start_containers
    fi

    if [[ $RESTART -eq 1 ]]; then
        restart_containers
    fi

    if [[ $STATUS -eq 1 ]]; then
        show_status
    fi

    if [[ $LOGS -eq 1 ]]; then
        show_logs "$LOG_SERVICE"
    fi

    # Change back to original directory after all tasks
    cd "$CURRENT_DIR"
}

# Error handling
trap 'error "Script interrupted"; cd "$CURRENT_DIR"; exit 1' INT TERM

# Run main function
main "$@"
