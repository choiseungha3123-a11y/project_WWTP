package kr.kro.prjectwwtp.domain;

import java.time.LocalDateTime;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Temporal;
import jakarta.persistence.TemporalType;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
@ToString
@Builder
@AllArgsConstructor
@NoArgsConstructor
@Entity
public class TmsOrigin {
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	private long tmsNo;
	@Temporal(TemporalType.TIMESTAMP)
	@Column(name = "time", updatable = false)
	private LocalDateTime tmsTime;
	private double toc;
	private double ph;
	private double ss;
	private int flux;
	private double tn;
	private double tp;
}
