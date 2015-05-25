
import os, sys;
import time;
from multiprocessing.queues import SimpleQueue;
from multiprocessing import Process;

numProcs = 6;
queue = SimpleQueue();
procs = [ ];

def worker(queue, name):
    while True:
        cmd = queue.get();
        if cmd == None:
            break;
        print name, cmd;
        os.system(cmd);

def main():

    global queue, procs, numProcs;

    if len(sys.argv) < 2:
        print "python dotasks.py task_list.txt";
        print "python dotasks.py procs:10 task_list.txt";
        return;

    inputFile =  "";
    for arg in sys.argv[1:]:
        if arg.startswith("procs:"):
            numProcs = int(arg.partition(":")[2]);
        else:
            inputFile = arg;

    if inputFile == "": return;

    with open(inputFile) as fin:
        for line in fin:
            line = line.strip();
            if line == "": continue;
            queue.put(line);

    for i in range(numProcs):
        queue.put(None);

    for i in range(numProcs):
        p = Process(target=worker, args=(queue, "worker-%d" % i));
        p.start();
        procs.append(p);

    while not queue.empty():
        time.sleep(10);

if __name__ == "__main__":
    main();
