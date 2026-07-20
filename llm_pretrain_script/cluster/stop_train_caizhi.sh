bash auto_fault_manager.sh --hostfile ../hostfile.runtime.128 --worldsize 128 --stop
kill -9 $(cat .auto_fault_manager.pid 2>/dev/null) 2>/dev/null
rm -f .auto_fault_manager*.pid
bash stop_all.sh ../hostfile.runtime.128
