import os
import matplotlib.pyplot as plt

IMAGEPATH = 'img/'

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

def plot_data(work, data, results):
    imgpath = IMAGEPATH + work + '.' + data + '.png'

    plt.clf()
    freqs = []
    itemdata = {}
    for freq in results:
        freqs.append(freq)
        for item in results[freq]:
            if item not in itemdata:
                itemdata[item] = []
            itemdata[item].append( results[freq][item] )

    for item in itemdata:
        label = ''
        alpha = 0.7
        if data == 'time':
            label = 'Core ' + str(item)
        elif data == 'power':
            label = 'Socket ' + str(item)
        elif item == -1:
            label = 'Mean'
            alpha = 1.0

        plt.plot(freqs, itemdata[item], label=label, color='k', alpha=alpha)

    plt.xlabel('Frequency (KHz)')
    plt.ylabel('Time (ms)' if data == 'time' else 'Power (W)')

    plt.savefig(imgpath)

def main():
    results = get_results()

    for work in results:
        for data in results[work]:
            plot_data(work, data, results[work][data])


if __name__ == '__main__':
    main()
