def optimizer_config(name_data):
    if name_data =='cifar10':
        learning_rate = 0.01
        momentum = 0.9
        weight_decay = 1e-4
        lr_step_size=1 # consider to change it to 5? (local round number)
        gamma=1
        milestones=[100]
        
    if name_data =='mnist':
        learning_rate = 0.01#0.001
        momentum = 0.9
        weight_decay = 0.001#1e-4 ,0.05
        lr_step_size=1#30 
        gamma=0.98
        milestones=[40,60,80] #not used currently
        
    if name_data=='fashion_mnist':
        learning_rate= 0.01 #0.002 # 2e-6
        momentum=0.9
        weight_decay=0.001#0.001#1e-4, 0.05
        lr_step_size=1#30
        gamma=0.98
        milestones=[100] #not used currently

    if name_data == 'emnist':
        learning_rate = 0.01 #0.002 # 2e-6
        momentum = 0.9
        weight_decay = 0.001
        lr_step_size=1
        gamma=0.98
        milestones=[100]
        
        #0.01*math.pow(0.98,100)=0.001326 (rationale)
        

    return learning_rate, momentum, weight_decay, lr_step_size, gamma, milestones