

class Task:
    def __init__(self,
                 model,
                 task,
                 train_lang,
                 data_num,
                 test_lang,
                 prompt_num,
                 epoch,
                 batch_size,
                 need_train):
        self.model = model
        self.task = task
        self.train_lang = train_lang
        self.test_lang = test_lang
        self.prompt_num = prompt_num
        self.epoch = epoch
        self.batch_size = batch_size
        if data_num == "test":
            data_num = 100
            self.epoch = 2
        self.data_num = data_num
        self.need_train = need_train

    def decide_gpu(self, id):
        self.gpu = id

    def calc_time(self):
        rounds = self.data_num * self.epoch // self.batch_size
        time = 0
        if not self.need_train:
            return 3*60
        if self.model == "codet5_base":
            time = rounds * 1.13
        elif self.model == "codebert":
            time = rounds * 3.6
        return int(time)

    def gen_cmd(self):
        cmd = "python run_exp.py"
        cmd += " --model_tag {}".format(self.model)
        cmd += " --task {}".format(self.task)
        cmd += " --train_lang {}".format(self.train_lang)
        cmd += " --data_num {}".format(self.data_num)
        cmd += " --test_lang {}".format(self.test_lang)
        cmd += " --prompt_num {}".format(self.prompt_num)
        cmd += " --model_dir model --res_dir res --summary_dir res"
        cmd += " --epoch {}".format(self.epoch)
        cmd += " --batch_size {}".format(self.batch_size)
        cmd += " --gpu {}".format(self.gpu)
        if self.need_train:
            cmd += " --need_train"
        return cmd


class TaskList:
    def __init__(self, gpu_num, use_gpu=-1):
        self.task_list = []
        self.gpu_num = gpu_num
        self.use_gpu = use_gpu

    def add_task(self, task):
        self.task_list.append(task)

    def cmd_use_all_gpu(self):
        total_time = 0
        for task in self.task_list:
            total_time += task.calc_time()

        split_time = total_time // self.gpu_num
        task_gpu_list = []

        task_id = 0
        for gpu in range(self.gpu_num):
            tasks = []
            gpu_time = 0
            while gpu_time < split_time and task_id < len(self.task_list):
                task = self.task_list[task_id]
                gpu_time += task.calc_time()
                task.decide_gpu(gpu)
                tasks.append(task)
                task_id += 1
            task_gpu_list.append(tasks)
            m, s = divmod(gpu_time, 60)
            h, m = divmod(m, 60)
            print("gpu: {}, time predict: {}:{}:{}".format(gpu, h, m, s))

        for gpu in range(self.gpu_num):
            tasks_cmd = ""
            for task in task_gpu_list[gpu]:
                tasks_cmd += task.gen_cmd() + "\n"
            with open("run_{}.sh".format(gpu), 'w') as f:
                f.write(tasks_cmd)
            print("nohup ./run_{}.sh > output/{}.log 2>&1 &".format(gpu, gpu))

    def cmd_use_single_gpu(self):
        task_list = []
        gpu_time = 0
        for task in self.task_list:
            gpu_time += task.calc_time()
            task.decide_gpu(self.use_gpu)
            task_list.append(task)
        m, s = divmod(gpu_time, 60)
        h, m = divmod(m, 60)
        print("time predict: {}:{}:{}".format(h, m, s))
        tasks_cmd = ""
        for task in task_list:
            tasks_cmd += task.gen_cmd() + "\n"
        with open("run_{}.sh".format(self.use_gpu), 'w') as f:
            f.write(tasks_cmd)
        print("nohup ./run_{}.sh > output/{}.log 2>&1 &".format(self.use_gpu, self.use_gpu))


    def generate_cmd(self):
        print("cd /mnt/sda/cn/codet5/sh")
        print("conda activate pretrain_cuinan")

        if self.use_gpu == -1:
            self.cmd_use_all_gpu()
        else:
            self.cmd_use_single_gpu()


if __name__ == "__main__":
    model_list = ["codebert", "codet5_base"]
    sum_langs = ["java", "python", "go", "ruby", "javascript", "php"]
    test_langs = ["go", "ruby", "javascript", "php"]
    size = [5000]
    prompt_num_list = [5, 0]
    gpu_num = 2

    task_list = TaskList(gpu_num, use_gpu=0)

    # task_list.add_task(
    #     Task(model=model_list[1], task="summarize", train_lang="java", data_num=5000, test_lang="java",
    #          prompt_num=5, epoch=10, batch_size=20, need_train=True))
    for test_lang in test_langs:
        task_list.add_task(
            Task(model=model_list[1], task="summarize", train_lang="java", data_num=1000, test_lang=test_lang,
                 prompt_num=15, epoch=10, batch_size=20, need_train=False))
    task_list.generate_cmd()
    # Best ppl
    # Finish and take
    # 873958 873880
