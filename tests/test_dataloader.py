from unittest import TestCase
from os.path import isfile
from src.lib.dataProcessing.data_processer import DataProcesser


class TestDataLoader(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        global dataloader

        db_path = 'tests/fixtures/sample_database'
        dataloader = DataProcesser(db_path, spectrum_size=1024)

    ##########################################################################################
    #  T E S T    C A S E S
    ##########################################################################################

    def test_load_dir(self):

        dataset = dataloader.load_spectra_from_dir(folder_nmb=1)

        self.assertIsInstance(dataset, dict)
        self.assertListEqual(list(dataset.keys()), ["30"])
        self.assertEqual(len(list(dataset.items())[0][1]), 2)


    def test_load_database(self):

        dataset = dataloader.load_spectra_from_database()

        self.assertListEqual(list(dataset.keys()), ["30", "12"])
        self.assertEqual(len(list(dataset.items())[0][1]), 3)
        self.assertEqual(len(list(dataset.items())[1][1]), 1)


    def test_save_and_load(self):
        
        saving_path = 'tests/fixtures/sample_database'
        _ = dataloader.load_spectra_from_database(saving_path)
        dataset = dataloader.load_spectra_from_file('tests/fixtures/sample_database/database.pickle')

        self.assertTrue(isfile('tests/fixtures/sample_database/database.pickle'))
        self.assertRaises(TypeError, dataloader.load_spectra_from_file)
        self.assertListEqual(list(dataset.keys()), ["30", "12"])
        self.assertEqual(len(list(dataset.items())[0][1]), 3)
        self.assertEqual(len(list(dataset.items())[1][1]), 1)
        