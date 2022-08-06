# Detect_Recip_Saw

PURPOSE AND DESCRIPTION:

This project is intended to use Digital Signal Processing to Detect Catalytic Converter Theft.  The project started after a neighbor’s catalytic converter was stolen from the apartment parking lot.  Once that happened, I started researching Digital Signal Processing and signal detection.

The Objective Signal of the project is from a battery-powered reciprocating saw operating at approximately 3,000 strokes per minute.  The environment is a busy (“noisy”) apartment parking lot.  In this environment, the objective signal is challenging to classify at varying ranges while other noises in the environment interfere with or resemble the saw’s signal.  

A couple of notes:
-	The work by Velardo, Lys, gabemagee, GianlucaPaolocci, and others has been very helpful to explore this issue
-	The version included with this post makes progress to characterize the objective signal; false alarms may still occur (and collected) with this version
-	Initial theories that looked for consistent events in the Objective Signal led to a high rate of false alarms; when I switched the approach to look for inconsistent events, the distinction between Other Signals and the Objective Signal became clearer

Read more in the pdf.
