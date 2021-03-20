<TeXmacs|1.99.19>

<style|generic>

<\body>
  <section|Offline <math|\<lambda\>>-Return Algorithm>

  (a)

  <\eqnarray*>
    <tformat|<table|<row|<cell|\<delta\><rsub|0>>|<cell|=>|<cell|<around*|(|R<rsub|0>+\<gamma\>V<around*|(|S<rsub|1>|)>|)><with|color|red|-V<around*|(|S<rsub|0>|)>>>>|<row|<cell|>|<cell|=>|<cell|
    <around*|(|-1+1\<times\><around*|(|-1|)>|)><with|color|red|-<around*|(|-3|)>>>>|<row|<cell|>|<cell|=>|<cell|1>>|<row|<cell|\<delta\><rsub|1>>|<cell|=>|<cell|<around*|(|R<rsub|1>+\<gamma\>V<around*|(|S<rsub|2>|)>|)><with|color|red|-V<around*|(|S<rsub|1>|)>>>>|<row|<cell|>|<cell|=>|<cell|<around*|(|-1+1\<times\><around*|(|0|)>|)><with|color|red|-<around*|(|-1|)>>>>|<row|<cell|>|<cell|=>|<cell|0>>>>
  </eqnarray*>

  (b)

  with <math|\<lambda\>=1> we recover the MC form of the update equation:

  <\eqnarray*>
    <tformat|<table|<row|<cell|V<around*|(|S<rsub|t>|)>>|<cell|\<leftarrow\>>|<cell|V<around*|(|S<rsub|t>|)>+\<alpha\><around*|(|G<rsub|t><rsup|\<lambda\>>-V<around*|(|S<rsub|t>|)>|)>>>>>
  </eqnarray*>

  then, for states <math|S<rsub|t>>, <math|t=<around*|{|0, 1|}>>, we have:

  <\eqnarray*>
    <tformat|<table|<row|<cell|V<around*|(|S<rsub|0>|)>>|<cell|\<leftarrow\>>|<cell|-3+0.5\<times\><around*|(|-2-<around*|(|-3|)>|)>>>|<row|<cell|>|<cell|\<leftarrow\>>|<cell|-2.5>>|<row|<cell|V<around*|(|S<rsub|1>|)>>|<cell|\<leftarrow\>>|<cell|-1+0.5\<times\><around*|(|-1-<around*|(|-1|)>|)>>>|<row|<cell|>|<cell|\<leftarrow\>>|<cell|-1>>>>
  </eqnarray*>

  <with|color|red|all other states remain unchanged.>

  \;

  <section|Implementation>

  <with|color|red|Note to self: don't forget to multiply <verbatim|(1-done)>
  when calculating TD error>
</body>

<\initial>
  <\collection>
    <associate|page-medium|automatic>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1|../../../../.TeXmacs/texts/scratch/no_name_14.tm>>
    <associate|auto-2|<tuple|2|?|../../../../.TeXmacs/texts/scratch/no_name_14.tm>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Offline
      <with|mode|<quote|math>|\<lambda\>>-Return Algorithm>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>