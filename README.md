# classification
A try to use pomegranate lib for prediction

This is a try to use pomegranate library to clasify sleep stages on simplified sleep data 
- eeg reduced to one chanel and then discretized by rounding.

It may help people, who get confused by what the input may look like according to pomegranate documentation.
Even though the autor states you can use any array like data, he sometimes means it really should be a list,
because in the source code of the library the comparision is sometimes done by "not equal to" and sometimes
by "!=", which is not applicable to arrays. 
