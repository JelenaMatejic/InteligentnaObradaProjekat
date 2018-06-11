import matplotlib.markers as mark

def listOfMarkers():
    # izdvojimo sve moguce markere
    markers = []
    for m in mark.MarkerStyle.markers:
        markers.append(m)
    markers = markers[:-4] # Jer su poslednja 4 markera None ili ' '
    return markers