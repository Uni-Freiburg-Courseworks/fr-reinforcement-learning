<TeXmacs|1.99.19>

<style|generic>

<\body>
  <section|Score Function>

  <\eqnarray*>
    <tformat|<table|<row|<cell|\<nabla\><rsub|\<theta\>> log
    \<pi\><around*|(|a=1\|s, \<theta\>|)>>|<cell|=>|<cell|\<nabla\><rsub|\<theta\>>
    log \<sigma\><around*|(|s<rsup|T>\<theta\>|)>>>|<row|<cell|>|<cell|=>|<cell|
    <frac|1|\<sigma\><around*|(|s<rsup|T>\<theta\>|)>>
    \<nabla\><rsub|\<theta\>> \<sigma\><around*|(|s<rsup|T>\<theta\>|)>>>|<row|<cell|>|<cell|=>|<cell|<frac|1|\<sigma\><around*|(|s<rsup|T>\<theta\>|)>>
    \<sigma\><around*|(|s<rsup|T>\<theta\>|)>
    <around*|(|1-\<sigma\><around*|(|s<rsup|T>\<theta\>|)>|)>s>>|<row|<cell|>|<cell|=>|<cell|<around*|(|1-\<sigma\><around*|(|s<rsup|T>\<theta\>|)>|)>s>>|<row|<cell|>|<cell|=>|<cell|\<pi\><around*|(|a=0\<divides\>s,
    \<theta\>|)>\<cdot\>s>>|<row|<cell|\<nabla\><rsub|\<theta\>> log
    \<pi\><around*|(|a=0\|s, \<theta\>|)>>|<cell|=>|<cell|\<nabla\><rsub|\<theta\>>
    log <around*|(|1-\<sigma\><around*|(|s<rsup|T>\<theta\>|)>|)>>>|<row|<cell|>|<cell|=>|<cell|<frac|1|1-\<sigma\><around*|(|s<rsup|T>\<theta\>|)>>
    \<nabla\><rsub|\<theta\>> <around*|[|1-\<sigma\><around*|(|s<rsup|T>\<theta\>|)>|]>>>|<row|<cell|>|<cell|=>|<cell|-<frac|\<sigma\><around*|(|s<rsup|T>\<theta\>|)>|1-\<sigma\><around*|(|s<rsup|T>\<theta\>|)>>
    <around*|(|1-\<sigma\><around*|(|s<rsup|T>\<theta\>|)>|)>s>>|<row|<cell|>|<cell|=>|<cell|-\<sigma\><around*|(|s<rsup|T>\<theta\>|)>s>>|<row|<cell|>|<cell|=>|<cell|-\<pi\><around*|(|a=1\<divides\>s,\<theta\>|)>\<cdot\>s>>>>
  </eqnarray*>

  <section|REINFORCE>

  See implementation.
</body>

<initial|<\collection>
</collection>>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?|../../../../.TeXmacs/texts/scratch/no_name_14.tm>>
    <associate|auto-2|<tuple|2|?|../../../../.TeXmacs/texts/scratch/no_name_14.tm>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Score
      Function> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>REINFORCE>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>