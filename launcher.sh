#!/bin/bash
# WeatherEdge Launcher
# Usage: ./launcher.sh

cd "$(dirname "$0")"
PIDFILE=".weatheredge.pid"
LOGFILE="logs/weatheredge.log"
VENV=".venv/bin/activate"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

is_running() {
    [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

get_status() {
    if is_running; then
        PID=$(cat "$PIDFILE")
        UPTIME=$(ps -o etime= -p "$PID" 2>/dev/null | xargs)
        echo -e "${GREEN}RUNNING${NC} (PID $PID, uptime: $UPTIME)"
    else
        echo -e "${RED}STOPPED${NC}"
    fi
}

get_stats() {
    source "$VENV" 2>/dev/null
    python3 -c "
from weatheredge.db.database import make_engine, get_session
from weatheredge.db.models import Trade, Market, Forecast
from sqlalchemy import select, func
engine = make_engine('sqlite:///weatheredge.db')
try:
    with get_session(engine) as s:
        trades = s.execute(select(func.count(Trade.id))).scalar() or 0
        wins = s.execute(select(func.count(Trade.id)).where(Trade.status=='won')).scalar() or 0
        losses = s.execute(select(func.count(Trade.id)).where(Trade.status=='lost')).scalar() or 0
        open_pos = s.execute(select(func.count(Trade.id)).where(Trade.status=='open')).scalar() or 0
        total_pnl = s.execute(select(func.sum(Trade.pnl))).scalar() or 0
        forecasts = s.execute(select(func.count(Forecast.id))).scalar() or 0
        wr = (wins/(wins+losses)*100) if (wins+losses) > 0 else 0
    print(f'  Trades: {trades} (open: {open_pos}, won: {wins}, lost: {losses})')
    print(f'  Win Rate: {wr:.1f}%')
    print(f'  Total P&L: \${total_pnl:.2f}')
    print(f'  Forecasts stored: {forecasts}')
except Exception as e:
    print(f'  DB not available: {e}')
" 2>/dev/null
}

show_header() {
    clear
    echo -e "${BOLD}${CYAN}"
    echo "  ╔═══════════════════════════════════════╗"
    echo "  ║       WeatherEdge Trading Bot         ║"
    echo "  ║       Paper Trading Mode              ║"
    echo "  ╚═══════════════════════════════════════╝${NC}"
    echo ""
    echo -e "  Status: $(get_status)"
    echo -e "  Mode:   ${YELLOW}paper${NC}"
    echo ""
    echo -e "${BOLD}  --- Stats ---${NC}"
    get_stats
    echo ""
    if [ -f "$LOGFILE" ]; then
        echo -e "${BOLD}  --- Last 5 log lines ---${NC}"
        tail -5 "$LOGFILE" 2>/dev/null | sed 's/^/  /'
        echo ""
    fi
}

do_start() {
    if is_running; then
        echo -e "  ${YELLOW}Already running (PID $(cat "$PIDFILE"))${NC}"
        return
    fi
    echo -e "  ${CYAN}Starting WeatherEdge...${NC}"
    source "$VENV"
    mkdir -p logs
    nohup python launch.py >> logs/stdout.log 2>&1 &
    echo $! > "$PIDFILE"
    sleep 2
    if is_running; then
        echo -e "  ${GREEN}Started (PID $(cat "$PIDFILE"))${NC}"
    else
        echo -e "  ${RED}Failed to start. Check logs/stdout.log${NC}"
        rm -f "$PIDFILE"
    fi
}

do_stop() {
    if ! is_running; then
        echo -e "  ${YELLOW}Not running${NC}"
        rm -f "$PIDFILE"
        return
    fi
    PID=$(cat "$PIDFILE")
    echo -e "  ${CYAN}Stopping (PID $PID)...${NC}"
    kill "$PID" 2>/dev/null
    sleep 2
    if is_running; then
        kill -9 "$PID" 2>/dev/null
    fi
    rm -f "$PIDFILE"
    echo -e "  ${GREEN}Stopped${NC}"
}

# Main menu loop
while true; do
    show_header
    echo -e "  ${BOLD}[1]${NC} Start    ${BOLD}[2]${NC} Stop    ${BOLD}[3]${NC} Logs    ${BOLD}[4]${NC} Refresh    ${BOLD}[q]${NC} Quit"
    echo ""
    read -rp "  > " choice
    case "$choice" in
        1) do_start; sleep 2 ;;
        2) do_stop; sleep 1 ;;
        3) echo -e "\n  ${CYAN}--- Live Logs (Ctrl+C to exit) ---${NC}\n"
           tail -f "$LOGFILE" 2>/dev/null | sed 's/^/  /'
           ;;
        4) continue ;;
        q|Q) echo -e "\n  ${CYAN}Ciao!${NC}\n"; exit 0 ;;
        *) echo -e "  ${RED}?${NC}"; sleep 1 ;;
    esac
done
