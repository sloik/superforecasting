------------------------- MODULE ForecastLifecycle -------------------------
EXTENDS Integers, Sequences, FiniteSets
CONSTANTS MaxIds, MaxEvents, Mutant
VARIABLES rows, frozen, seals, events
vars == <<rows, frozen, seals, events>>
Ids == 1..MaxIds
Probabilities == 0..2 \* scaled by two; Brier scores scaled by four
Empty == [status |-> "absent", p |-> 0, history |-> <<>>, score |-> -1, outcome |-> -1]
Init == /\ rows = [i \in Ids |-> Empty]
        /\ frozen = [i \in Ids |-> FALSE]
        /\ seals = rows
        /\ events = <<>>
Created == {i \in Ids : rows[i].status # "absent"}
Room == Len(events) < MaxEvents
Record(i, kind, p, o) == Append(events, [id |-> i, kind |-> kind, p |-> p, outcome |-> o])
Add == \E p \in Probabilities:
    LET i == IF Mutant = "ids" /\ Created # {} THEN 1 ELSE Cardinality(Created) + 1 IN
    /\ Room /\ i \in Ids
    /\ rows' = [rows EXCEPT ![i] = [Empty EXCEPT !.status = "open", !.p = p, !.history = <<p>>]]
    /\ events' = Record(i, "add", p, -1)
    /\ UNCHANGED <<frozen, seals>>
Update == \E i \in Ids, p \in Probabilities:
    /\ Room /\ (rows[i].status = "open" \/ (Mutant = "terminal" /\ frozen[i]))
    /\ rows' = [rows EXCEPT ![i].p = p,
                  ![i].history = IF Mutant = "history" THEN <<p>> ELSE Append(@, p)]
    /\ events' = Record(i, "update", p, -1)
    /\ UNCHANGED <<frozen, seals>>
Resolve == \E i \in Ids, o \in 0..1:
    LET p == IF Mutant = "score" THEN Head(rows[i].history) ELSE rows[i].p
        value == (p - 2*o) * (p - 2*o)
        row == [rows[i] EXCEPT !.status = "resolved", !.score = value, !.outcome = o]
    IN /\ Room /\ rows[i].status = "open"
       /\ rows' = [rows EXCEPT ![i] = row]
       /\ frozen' = [frozen EXCEPT ![i] = TRUE]
       /\ seals' = [seals EXCEPT ![i] = row]
       /\ events' = Record(i, "resolve", rows[i].p, o)
Supersede == \E i \in Ids:
    LET row == [rows[i] EXCEPT !.status = "superseded",
                   !.score = IF Mutant = "unscored" THEN 0 ELSE @] IN
    /\ Room /\ rows[i].status = "open"
    /\ rows' = [rows EXCEPT ![i] = row]
    /\ frozen' = [frozen EXCEPT ![i] = TRUE]
    /\ seals' = [seals EXCEPT ![i] = row]
    /\ events' = Record(i, "supersede", rows[i].p, -1)
Next == Add \/ Update \/ Resolve \/ Supersede
Spec == Init /\ [][Next]_vars
TypeOK == /\ rows \in [Ids -> [status : {"absent", "open", "resolved", "superseded"},
                    p : Probabilities, history : Seq(Probabilities), score : -1..4, outcome : -1..1]]
          /\ frozen \in [Ids -> BOOLEAN]
          /\ seals \in [Ids -> [status : {"absent", "open", "resolved", "superseded"},
                    p : Probabilities, history : Seq(Probabilities), score : -1..4, outcome : -1..1]]
          /\ Len(events) <= MaxEvents
TerminalImmutable == \A i \in Ids: frozen[i] => rows[i] = seals[i]
HistoryComplete == \A i \in Created:
    /\ Len(rows[i].history) = Cardinality({n \in 1..Len(events):
            events[n].id = i /\ events[n].kind \in {"add", "update"}})
    /\ rows[i].p = rows[i].history[Len(rows[i].history)]
LatestProbabilityScore == \A i \in Ids: rows[i].status = "resolved" =>
    rows[i].score = (rows[i].p - 2*rows[i].outcome) * (rows[i].p - 2*rows[i].outcome)
OnlyResolvedScored == \A i \in Ids: (rows[i].status = "resolved") <=> (rows[i].score # -1)
UniqueIds == Cardinality(Created) = Cardinality({n \in 1..Len(events): events[n].kind = "add"})
=============================================================================
