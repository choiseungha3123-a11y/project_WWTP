package kr.kro.prjectwwtp.domain;

import java.time.LocalDateTime;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Temporal;
import jakarta.persistence.TemporalType;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

/**
 * 메모 전송용 DTO
 * - 화면(또는 API) <-> 컨트롤러 간에 메모 데이터를 주고받을 때 사용합니다.
 */
@Getter
@Setter
@ToString
@Builder
@AllArgsConstructor
@NoArgsConstructor
@Entity
public class Memo{
    @Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
    private long memoNo;
    private String content;
    
    @ManyToOne
    @JoinColumn(name="createUserNo")
    private Member createMember;
    @ManyToOne
    @JoinColumn(name="diableUseNo")
    private Member disableMember;
    
    @Temporal(TemporalType.TIMESTAMP)
	LocalDateTime createTime;
    @Temporal(TemporalType.TIMESTAMP)
	LocalDateTime disableTime;
}
