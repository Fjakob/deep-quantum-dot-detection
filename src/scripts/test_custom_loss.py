from config.imports import *


def signal_window(x, idx, shift):
    idx_left = idx - shift
    idx_right = idx + shift + 1

    # take signal window
    if idx_left < 0:
        padding_left = np.zeros( abs(idx_left) )
        x_window = np.concatenate( (padding_left, x[0:idx_right]) )
    elif idx_right > len(x):
        padding_right = np.zeros( idx_right - len(x) )
        x_window = np.concatenate( (x[idx_left:len(x)], padding_right) )
    else:
        x_window = x[idx_left:idx_right]

    return x_window


def shape_loss(x1, x2, window_size=3, exploration_space=2):
    n = len(x1)
    assert n == len(x2)

    shift = int((window_size-1)/2)

    errors = []
    for idx in range(n):
        local_errors = []
        x1_window = signal_window(x1, idx, shift)
        x2_window = signal_window(x2, idx, shift)
        local_errors.append((np.linalg.norm(x1_window-x2_window)**2)/len(x1_window))
        for jdx in range(1, exploration_space+1):
            x2_window_up = signal_window(x2, idx+jdx, shift)
            x2_window_down = signal_window(x2, idx-jdx, shift)
            local_errors.append((np.linalg.norm(x1_window-x2_window_up)**2)/len(x1_window))
            local_errors.append((np.linalg.norm(x1_window-x2_window_down)**2)/len(x1_window))
        e = np.min(local_errors)
        errors.append(e)
    return np.mean(errors)


def window_loss(x1, x2, window_size=5):
    diff = x1 - x2
    shift = int((window_size-1)/2)

    e = []
    for idx in range(len(diff)):
        window = signal_window(diff, idx, shift)
        e.append(np.mean(window))

    return np.linalg.norm(e)



        
        
if __name__ == "__main__":

    target = np.asarray([0,0,0, 0, 1, 0, 0, 0, 0, 2, 10, 1, 0, 0, 0, 0])

    x1     = np.asarray([0,0,0, 0, 0, 0, 0, 0, 2, 10, 1, 0, 0, 0, 0, 0])
    x2     = np.asarray([0,0,0, 0, 0, 0, 0, 0, 0, 2, 10, 1, 0, 0, 0, 0])
    x3     = np.asarray([0,0,0, 0, 0, 0, 0, 0, 0, 1, 5, 1, 0, 0, 0, 0])

    print(f"Error to x1: {(np.linalg.norm(x1-target)**2)/len(x1)}")
    print(f"Error to x2: {(np.linalg.norm(x2-target)**2)/len(x2)}")
    print(f"Error to x3: {(np.linalg.norm(x3-target)**2)/len(x3)}\n")

    print(f"Shape error to x1: {shape_loss(target,x1)}")
    print(f"Shape error to x2: {shape_loss(target,x2)}") 
    print(f"Shape error to x3: {shape_loss(target,x3)}\n")

    print(f"Window error to x1: {window_loss(x1,target)}")
    print(f"Window error to x2: {window_loss(x2,target)}") 
    print(f"Window error to x3: {window_loss(x3,target)}")

    plt.figure(figsize=(6,8))
    plt.subplot(3,1,1)
    plt.plot(target, label='target')
    plt.plot(x1, '--', label='reconstruction')
    plt.ylim((0,10))
    plt.legend()
    plt.subplot(3,1,2)
    plt.plot(target, label='target')
    plt.plot(x2, '--', label='reconstruction')
    plt.ylim((0,10))
    plt.legend()
    plt.subplot(3,1,3)
    plt.plot(target, label='target')
    plt.plot(x3, '--', label='reconstruction')
    plt.ylim((0,10))
    plt.legend()
    plt.show()