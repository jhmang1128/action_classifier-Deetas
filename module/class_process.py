
###################################################################################################################
# handling action class
###################################################################################################################
def check_act_class(action_class):
    if action_class == 'Going':
        return False
    elif action_class == 'Coming':
        return False
    elif action_class == 'Crossing':
        return False
    elif action_class == 'Stopping':
        return False
    elif action_class == 'Moving': # not use
        return True
    elif action_class == 'Stoping': # not use
        return True
    elif action_class == 'Avoiding': # not use
        return True
    elif action_class == 'Opening': # not use
        return True
    elif action_class == 'Closing': # not use
        return True
    elif action_class == 'On': # not use
        return True
    elif action_class == 'Off': # not use
        return True

def convert_action_class (action_class):
    ### output = 0
    if action_class == 'Going':
        output = 0
    elif action_class == 'Coming':
        output = 1
    elif action_class == 'Crossing':
        output = 2
    elif action_class == 'Stopping':
        output = 3
    elif action_class == 'Moving': # not use
        output = 4
    elif action_class == 'Stoping': # not use
        output = 5
    elif action_class == 'Avoiding': # not use
        output = 6
    elif action_class == 'Opening': # not use
        output = 7
    elif action_class == 'Closing': # not use
        output = 8
    elif action_class == 'On': # not use
        output = 9
    elif action_class == 'Off': # not use
        output = 10

    return output