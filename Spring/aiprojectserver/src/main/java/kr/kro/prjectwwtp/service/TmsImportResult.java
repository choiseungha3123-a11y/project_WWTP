package kr.kro.prjectwwtp.service;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class TmsImportResult {
    private int totalLines;
    private int savedCount;
    private int skippedEmptyOrComment;
    private int skippedMalformed;
    private int skippedParseError;
    private boolean detectedHeader;
}
