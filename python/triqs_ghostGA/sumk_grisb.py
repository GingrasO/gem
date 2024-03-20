from triqs_dft_tools.sumk_dft import *

class SumkGRISB(SumkDFT):
      '''
      Inherent from SumkDFT for GRISB k-summation
      '''
      def ksum1(self):
          '''
          The first k-summation for density matrix
          '''
          pass

      def ksum2(self):
          '''
          The second k-sum for kinetic energy
          '''
          pass

      def calc_mu(self):
          '''
          Override the sumk calc_mu for GRISB
          '''
          pass

      def calc_density_correction(self):
          '''
          Overide the density correction for GRISB
          '''
          pass
