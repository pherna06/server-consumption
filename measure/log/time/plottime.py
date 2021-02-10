import os
import matplotlib.pyplot as plt

# TODO: refactor to just handle time plot OR implement general utility 'plotdata.py'

IMAGEPATH = 'img/'
STYLES = {
        'intproduct': 'r-',
        'floatproduct': 'r:',
        'inttranspose': 'b-',
        'floattranspose': 'b:',
        'intscalar': 'g-',
        'floatscalar': 'g:',
        'intsort': 'y-',
        'floatsort': 'y:'
        }

def get_file_data(csv):
    csvf = open(csv, 'r')
    csvflines = csvf.readlines()
    csvf.close()

    results = {}

    dataread = False
    while csvflines:
        if not dataread:
            # Frequency
            freqstr = csvflines.pop(0)
            freq = int(freqstr)
            results[freq] = {}
            dataread = True
        else:
            # Data
            datastr = csvflines.pop(0)
            datastr = datastr.split(', ')
            item = int(datastr[0])
            data = float(datastr[1])

            results[freq][item] = data

            if item == -1:
                dataread = False

    return results

def get_results():
    csvfiles = [f for f in os.listdir() if f.split('.')[-1] == 'csv']

    results = {}
    for csv in csvfiles:
        words = csv.split('.')
        work = words[0]
        data = words[1]

        if work not in results:
            results[work] = {}

        results[work][data] = get_file_data(csv)

    return results

def format_results(results):
    formatted = {}

    for work in results:
        formatted[work] = {}
        for data in results[work]:
            formatted[work][data] = {}
            for freq in results[work][data]:
                for item in results[work][data][freq]:
                    if item not in formatted[work][data]:
                        formatted[work][data][item] = {}
                    formatted[work][data][item][freq] = results[work][data][freq][item]

    return formatted


def plot_items(work, data, results):
    imgpath = IMAGEPATH + work + '.' + data + '.png'
    plt.clf()

    for item in results:
        label = ''
        if data == 'time':
            label = 'Core ' + str(item)
        elif data == 'power':
            label = 'Socket ' + str(item)
        if item == -1:
            label = 'Mean'

        alpha = 0.3
        color = 'k'
        linestyle = ':'
        if item == -1:
            alpha = 1.0
            color = 'b'
            linestyle = '-'

        x, y = zip(*results[item].items())
        plt.plot(x, y, label=label, color=color, alpha=alpha, linestyle=linestyle)

    plt.xlabel('Frequency (KHz)')
    plt.ylabel('Time (ms)' if data == 'time' else 'Power (W)')

    plt.savefig(imgpath)

def plot_mean(data, results):
    imgpath = IMAGEPATH + 'mean.' + data + '.png'
    plt.clf()
    
    for work in results:
        label = work
        
        x, y = zip(*results[work][data][-1].items())
        plt.plot(x, y, STYLES[work], label=label)

    plt.xlabel('Frequency (KHz)')
    plt.ylabel('Time (ms)' if data == 'time' else 'Power (w)')
    plt.legend()

    plt.savefig(imgpath)

def main():
    results = get_results()
    results = format_results(results)

    for work in results:
        for data in results[work]:
            plot_items(work, data, results[work][data])

    datas = []
    for work in results:
        for data in results[work]:
            datas.append(data)
        break

    for data in datas:
        plot_mean(data, results)


if __name__ == '__main__':
    main()
