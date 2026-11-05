import numpy as np
import os

class Logger():
    def __init__(self, out_path, n_task, task_dict, test_sizes) -> None:
        self.acc_taw = np.zeros((n_task, n_task))
        self.forg_taw = np.zeros((n_task, n_task))
        
        self.acc_tag = np.zeros((n_task, n_task))
        self.forg_tag  = np.zeros((n_task, n_task))

        self.perstep_acc_taw = np.zeros((n_task, n_task))
        self.perstep_acc_tag = np.zeros((n_task, n_task))
 
        self.task_len  =  [len(item) for item in task_dict.values()]
        self.test_sizes = test_sizes

        self.out_path = os.path.join(out_path, "logger")
        


    def update_accuracy(self, current_training_task_id, test_id, acc_taw_value, acc_tag_value):
        # accuracy matrix taw aware (yes task id), used to compute forgetting
        self.acc_taw[current_training_task_id, test_id] = acc_taw_value * 100
        # accuracy matrix task agnostic (no task id), used to compute forgetting
        self.acc_tag[current_training_task_id, test_id] = acc_tag_value * 100
        # used to compute the last per step accuracy task aware and task agnostic (weighted with the number of samples per class)
        self.perstep_acc_taw[current_training_task_id, test_id] =  (acc_taw_value*100) *  self.task_len[test_id]
        self.perstep_acc_tag[current_training_task_id, test_id] = (acc_tag_value*100) *  self.task_len[test_id] 





    def update_forgetting(self, current_training_task_id, test_id):
        # forgetting task aware and task agnostic
        self.forg_taw[current_training_task_id, test_id] = self.acc_taw[:current_training_task_id, test_id].max(0) - self.acc_taw[current_training_task_id, test_id]
        self.forg_tag[current_training_task_id, test_id] = self.acc_tag[:current_training_task_id, test_id].max(0) - self.acc_tag[current_training_task_id, test_id]
  

    def print_latest(self, current_training_task_id, test_id):
        print('\n >>> Test on task {:2d} : TAw acc={:5.1f}%, forg={:5.1f}%'
              ' | TAg acc ={:5.1f}%, forg ={:5.1f}%  <<<'.format(test_id, 
                                                                self.acc_taw[current_training_task_id, test_id],  self.forg_taw[current_training_task_id, test_id],
                                                                self.acc_tag[current_training_task_id, test_id],  self.forg_tag[current_training_task_id, test_id]))


    def compute_average(self):
        self.list_perstep_acc_taw = self.perstep_acc_taw.sum(1) / (np.tril(np.array(self.task_len))).sum(1)
        self.list_perstep_acc_tag = self.perstep_acc_tag.sum(1) / (np.tril(np.array(self.task_len))).sum(1)
        
          
    def print_file(self):
        # save matrices taw and tag accuracies 
        np.savetxt(os.path.join(self.out_path, 'acc_taw.out'), self.acc_taw, delimiter=',', fmt='%.3f')
        np.savetxt(os.path.join(self.out_path, 'acc_tag.out'), self.acc_tag, delimiter=',', fmt='%.3f')
        # save matrices taw and tag forgetting   
        np.savetxt(os.path.join(self.out_path, 'forg_taw.out'), self.forg_taw, delimiter=',', fmt='%.3f')       
        np.savetxt(os.path.join(self.out_path, 'forg_tag.out'), self.forg_tag, delimiter=',', fmt='%.3f')
 
        # Save per-step incremental accuracy.
        np.savetxt(os.path.join(self.out_path, 'acc_taw_perstep.out'), self.list_perstep_acc_taw, delimiter=',', fmt='%.3f')     
        np.savetxt(os.path.join(self.out_path, 'acc_tag_perstep.out'),  self.list_perstep_acc_tag, delimiter=',', fmt='%.3f')


class FileOutputDuplicator:
    def __init__(self, duplicate, file):
        self.duplicate = duplicate
        self.file = file

    def write(self, data):
        self.duplicate.write(data)
        return self.file.write(data)

    def flush(self):
        self.duplicate.flush()
        self.file.flush()
