'''
Poniższy plik zawiera:
    - Klasy:
        a) VideoAnalyzer - kompleksowa klasa służąca do detekcji obiektów sparametryzowanych w metodzie __init__().
                           Obikety wykrywane są poprzez egzekwowanie poniższego algorytmu:
                            1. Maska Thresh filtrująca wartości z zakresu (100,100,100) - (255,255,255)
                            2. Etykietowanie
                            3. Filtracja po argumencie BoundingBoxArea
                            4. Filtracja po argumencie moments_hu - Niezmienniki pomagają wykryć poprawne elmenty
                            5. Filtracja poprzez odcięcie zadeklarowanych przez użytkownika kolorów
                Metody:
                    - __init__  - deklaracja właściwości, parametryzacja
                    - run - główna pętla procesu
                    - read_keyboard - interakcja z użytownikiem. Spacja symuluje wywołanie awaryjnego zatrzymania
                    - stop_running - procedura zatrzymania odczytu. Zamknięcie okien, czyszczenie procesu
                    - create_raport - tworzenie raportu  w formacie txt
                    - preprocess_img - pierwszy etap algorytmu wykrywania obiektów, filtorwanie obrazu
                    - position_and_size_filter - odfiltrowanie mniejszych obiektów, detekcja obiektu dopiero
                                                po przekroczeniu jego centroida za 1/5 odległości odczytywanego obrazu
                    - classify_obj - klasyfikacja ze względu na kolor, poprawność, element
                    - show_camera - wyśweitlanie obrazu z odczytanego pliku
                    - show_img - wyświetlenie obrazu. Wykorzystane przy wyświetleniu ostatniego zidentyfkowanego obiektu
                    - draw_objects_on_img - rysowanie ram zidentyfikowanego obiektu na obrazie

        b) LetterClass - Klasa przechowująca parametry identyfikacji danego obiektu
                Metody:
                    - in_area_bands - Zwraca prawdę jeśli argument(powierzchnia pokrycia obiektu identyfikowanego)
                                      mieści się w maksymalnych ramach klasy identyfikowanej
                    - in_moments_hu_bands - Zwraca prawdę jeśli argument(niezmienniki momentowe obiektu identyfikowanego)
                                            mieści się w maksymalnych ramach klasy identyfikowanej

                    - get_area_bands - zwraca maksymalne ramy powierzchni pokrycia klasy identyfikowanej
                    - get_moments_hu_bands - zwraca maksymalne ramy niezmienników momentu klasy identyfikowanej
    - Dekorator :
        - detected_letter_wrapper - Wyświetla w linii komend, w czasie rzeczywistym,
                                    wykryte obiekty na linii produkcykjnej. Wartości zostają przekazane
                                    z metody VideoAnalyzer.classify_obj()

'''
import numpy as np
import copy
import cv2
import skimage.measure as sime
import datetime


# Dekorator użyty na metodzie VideoAnalyzer.classify_obj()
def detected_letter_wrapper(func):
    '''Dekorator wyświetaljący parametry ostatniego wykrytego obiektu.'''

    def wrap(*args, **kwargs):
        result = func(*args, **kwargs)
        print(
            f"Rozpoznany znak nr {result['no']}: <{result['Letter']}> "
            f"{'prawidłowe' if result['Quality'] else 'wadliwe'}, kolor: {result['Color']} ")
        return result

    return wrap


class VideoAnalyzer:
    def __init__(self, wideo):
        ##########################################################################
        # Parametry identyfikacji obiektów
        ##########################################################################

        # Minialny próg odcięcia małych obiektów
        self.minimum_bounding_box_area = 1000

        # Wyszukiwane znaki
        self.letters_name = ['G', 'B']

        # Właściwość przechowująca ramy niezbędne do identyfikacji Liter
        self.letter_g = LetterClass(self.letters_name[0],
                                    [1750, 1850],
                                    [[0.448513, 0.582788], [0.014425, 0.019406], [0.008648, 0.022219],
                                     [0.001713, 0.004304], [0.000003, 0.000018], [-0.000241, -0.000050],
                                     [0.000011, 0.000038]])
        self.letter_b = LetterClass(self.letters_name[1],
                                    [1390, 1470],
                                    [[3.709769e-01, 3.939e-01], [1.93e-02, 2.656197e-02], [4.000418e-04, 5.903602e-04],
                                     [1.451022e-05, 3.737761e-05], [-2.920091e-09, 2.664241e-09],
                                     [-1.143801e-05, 4.956973e-06],
                                     [-4.912235e-09, 1.762891e-09]])
        # Wyszukwane litery. Przypisanie Litery do parametrów ją opisujących
        self.letters = {
            self.letters_name[0]: self.letter_g,
            self.letters_name[1]: self.letter_b
        }

        # Kolory - definicja ram i słowny opis będący kluczem
        self.color_bound = {
            'zolty': [(100, 100, 0), (200, 200, 30)],
            'zielony': [(0, 100, 0), (50, 255, 50)],
            'czerwony': [(0, 0, 0), (30, 30, 255)],
        }
        ##########################################################################
        # Deklaracja właściwości klasy
        ##########################################################################
        self.wideo_path = wideo
        self.wideo = cv2.VideoCapture(self.wideo_path)
        self.emergency_stop = False
        self.props = None
        self.img = None
        self.idx_filtered = None
        self.found_objects = []
        self.prev_obj_amount = 0
        self.obj_amount = 0
        self.opoznienie = 0
        self.lamp_kolor = (0, 0, 0)

    def run(self):
        while self.wideo.isOpened():
            if not self.emergency_stop:
                is_read_success, self.img = self.wideo.read()
                if is_read_success:
                    b = self.preprocess_img()
                    labels = sime.label(b)
                    self.props = sime.regionprops(labels)
                    self.position_and_size_filter()
                    self.obj_amount = len(self.idx_filtered)
                    if self.obj_amount > self.prev_obj_amount:
                        self.classify_obj()
                        self.opoznienie = 10
                    elif self.opoznienie > 0:
                        self.opoznienie -= 1
                    else:
                        self.lamp_kolor = (255, 255, 255)

                    self.show_camera()
                else:
                    self.stop_running()
            self.read_keyboard()
            self.prev_obj_amount = self.obj_amount

    def read_keyboard(self):
        k = cv2.waitKey(2) & 0xFF
        if k == 27:
            self.stop_running()
        if k == ord(' '):
            self.emergency_stop = not self.emergency_stop

    def stop_running(self):
        self.wideo.release()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def create_raport(self, file_name) -> None:
        found_objects = np.asarray([(d['Letter'], d['Quality'], d['Color']) for d in self.found_objects],
                                   dtype="object")

        with open(f"{file_name}.txt", "w") as file:
            file.write(f"Raport z dnia {datetime.datetime.now()}\n\n")
            file.write('-' * 100 + '\n')
            file.write(f"Zidentyfikowanych obiektów: {len(found_objects)}\n")

            for letter in self.letters_name:
                file.write('-' * 100 + '\n')
                file.write(f"Znak :  {letter}\n")

                data = [[' ', 'Poprawne', 'Wadliwe']]
                for color in self.color_bound:
                    popr = np.sum((found_objects[:, 0] == letter) & (found_objects[:, 1] == True) &
                                  (found_objects[:, 2] == color))
                    wad = np.sum((found_objects[:, 0] == letter) & (found_objects[:, 1] == False) &
                                 (found_objects[:, 2] == color))
                    data.append([color, popr, wad])

                # Zapis tabeli
                for row in data:
                    file.write("{:<10} {:<10} {:<10}\n".format(*row))

    def preprocess_img(self):
        min_val = 100
        b = np.logical_or(self.img[:, :, 0] > min_val, self.img[:, :, 1] > min_val)
        b = np.logical_or(b, self.img[:, :, 2] > min_val)
        return b

    def position_and_size_filter(self):
        img = copy.copy(self.img)
        self.idx_filtered = list(
            filter(lambda i: self.props[i]['BoundingBoxArea'] >= self.minimum_bounding_box_area and
                             self.props[i].centroid[0] > img.shape[1] // 5,
                   range(len(self.props))))

    @detected_letter_wrapper
    def classify_obj(self):
        img = copy.copy(self.img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Granice obiektu
        minr, minc, maxr, maxc = map(int, self.props[np.min(self.idx_filtered)].bbox)
        sliced_img = img_rgb[minr:maxr, minc:maxc]
        # Wyświetlenie ostatnio wykrtyego obiektu
        self.show_img(cv2.cvtColor(sliced_img,cv2.COLOR_BGR2RGB), 'Ostatni znak')

        # Pokrywana powierzchnia
        area = self.props[np.min(self.idx_filtered)]['BoundingBoxArea']
        # Niezmienniki momentowe
        moments = self.props[np.min(self.idx_filtered)]['moments_hu']

        max_color_sum = 0
        letter = None
        color = None
        ok = False

        # Detekcja znaku
        for k in self.letters:
            if self.letters[k].in_area_bands(area):
                letter = k
                ok = self.letters[k].in_moments_hu_bands(moments)

        # Detekcja koloru
        color_bound_values = list(self.color_bound.values())
        for i in range(len(color_bound_values)):
            lower_bound, upper_bound = color_bound_values[i]
            bin = cv2.inRange(sliced_img, np.array(lower_bound), np.array(upper_bound))
            result = cv2.bitwise_and(sliced_img, sliced_img, mask=bin)
            color_sum = np.sum(result)

            if color_sum > max_color_sum:
                max_color_sum = color_sum
                color = list(self.color_bound.keys())[i]

        # na podstawie odczytanej jakości zmiana koloru lampki na wizualizacji
        self.lamp_kolor = (0, 255, 0) if ok else (0, 0, 255)

        # Zapis zidentyfikowanego obiektu
        data = {
            'no': len(self.found_objects),
            'Letter': letter,
            'Quality': ok,
            'Color': color
        }
        self.found_objects.append(data)
        return data

    def show_camera(self):
        obraz_wynikowy = copy.copy(self.img)
        cv2.imshow('ramka', self.img)
        cv2.circle(obraz_wynikowy, (20, 15), 7, self.lamp_kolor, thickness=5, lineType=3, shift=0)
        self.draw_objects_on_img(obraz_wynikowy, [self.props[x] for x in self.idx_filtered])
        cv2.imshow('Wynik Analizy Obrazu', obraz_wynikowy)

    @staticmethod
    def show_img(img, name):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)  # Ustawia normalny tryb okna, który można zmieniać rozmiar ręcznie
        cv2.resizeWindow(name, 300, 300)  # Ustaw rozmiar okna na szerokość: 800, wysokość: 600

        # Wyświetl obraz
        cv2.imshow(name, img)

    @staticmethod
    def draw_objects_on_img(img, prop):
        for obiekt in prop:
            y0, x0 = map(int, obiekt.centroid)
            cv2.circle(img, (x0, y0), 5, (0, 0, 255), -1)  # Rysowanie punktu centralnego

            minr, minc, maxr, maxc = map(int, obiekt.bbox)
            cv2.rectangle(img, (minc, minr), (maxc, maxr), (0, 255, 255), 1)  # Rysowanie prostokąta


class LetterClass:
    def __init__(self, name, area_bands, moments_hu_bands):
        self.name = name
        self.area_bands = area_bands
        self.moments_hu_bands = moments_hu_bands

    def in_area_bands(self, obj):
        return self.area_bands[0] <= obj <= self.area_bands[1]

    def in_moments_hu_bands(self, obj):
        result = all(self.moments_hu_bands[i][0] <= obj[i] <=
                     self.moments_hu_bands[i][1] for i in range(len(obj)))
        return result

    def get_area_bands(self):
        return self.area_bands

    def get_moments_hu_bands(self):
        return self.moments_hu_bands
