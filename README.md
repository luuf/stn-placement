# stn-placement

These are experiments designed to figure out whether it's reasonable to place a spatial transformer network (stn) anywhere inside a neural network.
Some experiments also test whether it's a good idea to share layers between the classification network and the localization network.

Parts of this were written in tensorflow, but the latest version only use pytorch. The transformer module in the tensorflow version (found in stn/tf) is taken from https://github.com/kevinzakka/spatial-transformer-network.
